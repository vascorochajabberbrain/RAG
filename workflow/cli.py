"""
Guided CLI for the RAG workflow. Run with: python -m workflow.cli
"""
from workflow.models import WorkflowState, Step, ChunkingConfig
from workflow.runner import run_step


def main():
    from QdrantTracker import QdrantTracker
    tracker = QdrantTracker()
    state = WorkflowState(tracker=tracker)

    menu = """
Select an action:
  1 - Create new RAG from scratch (full pipeline)
  2 - Add source to existing collection
  3 - Run chunking/grouping only (you have raw text or collection)
  4 - Test Q&A on a collection
  5 - Show workflow status / list collections
  q - Quit
"""
    while True:
        action = input(menu).strip().lower()
        if action == "q":
            break
        if action == "1":
            _run_new_rag(state)
        elif action == "2":
            _add_source_to_collection(state)
        elif action == "3":
            _run_chunking_only(state)
        elif action == "4":
            _test_qa(state)
        elif action == "5":
            _show_status(state)
        else:
            print("Invalid option.")


def _run_new_rag(state: WorkflowState):
    state.collection_name = input("Collection name: ").strip() or state.collection_name
    if not state.collection_name:
        print("Collection name is required.")
        return
    use_grouping = input("Run grouping after chunking? (y/n) [n]: ").strip().lower() == "y"
    state.grouping_enabled = use_grouping

    print("\n--- Step: CREATE_COLLECTION ---")
    msg = run_step(state, Step.CREATE_COLLECTION)
    print(msg)

    stype = input("Source type (pdf / txt / url / csv) [pdf]: ").strip().lower() or "pdf"
    state.source_type = stype
    state.source_config = {}

    if stype == "pdf":
        path = input("Path to PDF file: ").strip()
        state.source_config = {"path": path, "source_label": path.split("/")[-1]}
    elif stype == "txt":
        path = input("Path to TXT file: ").strip()
        state.source_config = {"path": path, "source_label": path.split("/")[-1]}
    elif stype == "url":
        scraper = input("Scraper name (e.g. peixefresco): ").strip() or "peixefresco"
        state.source_config = {"scraper_name": scraper, "source_label": scraper}
    elif stype == "csv":
        path = input("Path to CSV file: ").strip()
        state.source_config = {"path": path, "source_label": path.split("/")[-1]}
    else:
        print("Unknown source type.")
        return

    print("\n--- Step: FETCH ---")
    msg = run_step(state, Step.FETCH)
    print(msg)
    if "Error" in msg:
        return

    if stype != "csv" and state.raw_text:
        do_clean = input("Run CLEAN step? (y/n) [n]: ").strip().lower() == "y"
        if do_clean:
            state.source_config["filter_name"] = input("Filter name (e.g. heyharper) or leave blank: ").strip() or "none"
            print(run_step(state, Step.CLEAN))

    if stype == "csv" and state.chunks:
        print("Chunks from CSV; skipping CHUNK.")
    else:
        use_default_chunk = input("Use default chunking? (y/n) [y]: ").strip().lower() != "n"
        if not use_default_chunk:
            suggest = input("Get chunking suggestion from AI? (y/n) [n]: ").strip().lower() == "y"
            if suggest:
                from workflow.suggest import suggest_chunking
                preview = (state.get_text_for_chunking() or "")[:1500]
                rec = suggest_chunking(preview, state.source_type or "unknown")
                state.chunking_config.batch_size = rec.get("batch_size", 10000)
                state.chunking_config.overlap_size = rec.get("overlap_size", 100)
                state.chunking_config.use_proposition_chunking = rec.get("use_proposition_chunking", True)
                print("Suggested:", rec)
            else:
                state.chunking_config.batch_size = int(input("Batch size [10000]: ") or "10000")
                state.chunking_config.overlap_size = int(input("Overlap [100]: ") or "100")
                state.chunking_config.use_proposition_chunking = input("Proposition-based (LLM) chunking? (y/n) [y]: ").strip().lower() != "n"
        print("\n--- Step: CHUNK ---")
        print(run_step(state, Step.CHUNK))

    if use_grouping and state.chunks:
        print("\n--- Step: GROUP ---")
        print(run_step(state, Step.GROUP))

    print("\n--- Step: PUSH_TO_QDRANT ---")
    print(run_step(state, Step.PUSH_TO_QDRANT))

    test = input("Ask a test question? (y/n) [n]: ").strip().lower() == "y"
    if test:
        state.source_config["test_question"] = input("Question: ").strip() or "What is this content about?"
        state.source_config["company_name"] = input("Company/assistant name (for prompt): ").strip() or "the assistant"
        print("\n--- Step: TEST_QA ---")
        print(run_step(state, Step.TEST_QA))


def _add_source_to_collection(state: WorkflowState):
    name = input("Existing collection name: ").strip()
    if not name:
        print("Collection name required.")
        return
    if name not in state.tracker.open_collections():
        state.collection_object = state.tracker.open(name)
    else:
        state.collection_object = state.tracker.get_collection(name)
    state.collection_name = name
    stype = input("Source type (pdf / txt / url / csv): ").strip().lower()
    state.source_type = stype
    state.source_config = {}
    if stype == "pdf":
        state.source_config = {"path": input("Path to PDF: ").strip()}
    elif stype == "txt":
        state.source_config = {"path": input("Path to TXT: ").strip()}
    elif stype == "url":
        state.source_config = {"scraper_name": input("Scraper name: ").strip() or "peixefresco"}
    elif stype == "csv":
        state.source_config = {"path": input("Path to CSV: ").strip()}
    else:
        print("Unknown source type.")
        return
    if state.source_config.get("path"):
        state.source_config["source_label"] = state.source_config["path"].split("/")[-1]
    elif state.source_config.get("scraper_name"):
        state.source_config["source_label"] = state.source_config["scraper_name"]

    print(run_step(state, Step.FETCH))
    if state.raw_text and stype != "csv":
        state.cleaned_text = state.raw_text
        print(run_step(state, Step.CHUNK))
    print(run_step(state, Step.PUSH_TO_QDRANT))


def _run_chunking_only(state: WorkflowState):
    print("Re-chunk or grouping: open the collection from the main menu (initial_menu) and use edit/ingest there, or run 'Create new RAG' and choose an existing collection when prompted.")


def _test_qa(state: WorkflowState):
    name = input("Collection name: ").strip()
    if not name:
        print("Collection name required.")
        return
    state.collection_name = name
    state.source_config = {
        "test_question": input("Question: ").strip() or "What is this content about?",
        "company_name": input("Company/assistant name: ").strip() or "Assistant",
    }
    print(run_step(state, Step.TEST_QA))


def _show_status(state: WorkflowState):
    print("Open collections:", state.tracker.open_collections())
    print("All collections:", state.tracker.all_collections())
    if state.collection_name:
        print("Current workflow collection:", state.collection_name)
    if state.chunks:
        print("Current chunks count:", len(state.chunks))


if __name__ == "__main__":
    main()
