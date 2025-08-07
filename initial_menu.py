import QdrantTracker


def main():
    qdrant_tracker = QdrantTracker.QdrantTracker()
    qdrant_tracker.all_collections()
    qdrant_tracker.open_collections()



if __name__ == '__main__':
    main()