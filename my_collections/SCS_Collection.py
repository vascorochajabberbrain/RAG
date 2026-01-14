from my_collections.Colletion import Collection
from objects.SCS import SCS

from ingestion.url_ingestion import main as url_ingestion
from llms.openai_utils import openai_chat_completion


class SCS_Collection(Collection):
    TYPE = "scs"
    """
    A class to represent a collection of self-contained sentences (SCS).
    """

    """------------------------Constructors------------------------"""
    def init_item_from_qdrant(self, point_data):
        return SCS.from_payload(point_data)

    """----------------------Public Methods------------------------"""
    
    def menu(self):
        menu = """Select an action:
        -- "q" to quit
        -- "app" to append a sentence
        -- "ins" to insert a sentence (by index)
        -- "del" to delete a sentence
        -- "mov" to move a sentence (by index)
        -- "i" to ingest (for now only url_ingestion)
        -- "p" to print the collection
        -- "s" to save the collection
"""
        action = input(menu)

        while action != "q":
            match action:
                case "app":
                    description = input("SCS:")
                    self.append_sentence(description)
                case "insert":
                    idx = int(input("Index:"))
                    description = input("SCS:")
                    self.insert_sentence(idx, description)
                case "del":
                    idx = int(input("Index:"))
                    self.delete_item(idx)
                case "mov":
                    from_idx = int(input("From Index:"))
                    to_idx = int(input("To Index:"))
                    self.move_item(from_idx, to_idx)
                case "i":
                    print("For now only url ingestion is supported")
                    chunks = url_ingestion()
                    self.append_sentences(chunks)
                case "p":
                    user_input = input("Press enter to print all or provide a list of indexes separated by spaces (example '1 2 3'):")
                    if user_input == "":
                        self.print()
                    else:
                        list_indexes = [int(i) for i in user_input.split()]
                        self.print(list_indexes)
                case "s":
                    return True
                case _:
                    print("Invalid action.")
            action = input(menu)
        return False

    """-------------SCS's------------"""

    def append_sentence(self, sentence, source=None):
        self.append_item(SCS(sentence, source))

    def append_sentences(self, sentences, source=None):
        if not isinstance(sentences, list):
            raise TypeError("Expected a list of sentences.")
        for sentence in sentences:
            if not isinstance(sentence, str):
                raise TypeError("Expected a string in the list of sentences.")
            self.append_item(SCS(sentence, source))

    def insert_sentence(self, idx, sentence, source=None):
        self.insert_item(idx, SCS(sentence, source))
    
    def get_sentence(self, idx):
        return self.get_item(idx).get_sentence()

    #do not like this, how we simply open the SCS object
    def get_all_scss(self):
        return self.items
    
    """-------------------Came from url_ingestion------------------"""
    """ #I don't think this should be here
    def create_batches_of_text(text, batch_size, overlap_size):
        batches = []
        start = 0
        while start < len(text):
            end = start + batch_size
            batch = text[start:end]
            batches.append(batch)
            start += batch_size - overlap_size
        return batches
"""
    #not in use for now
    def add_context(chunk, text):
        prompt = """You are a helpful assistant that rewrites sentences to include context from a larger text, without adding any new information.
        The goal is to make the sentence more understandable on its own, because it is going to be used for generating embeddings.
        Note, I don't want to include any more information than the one that makes it so you do not need to know anything else to understand it.
        You will receive the original sentence and the larger text. Please answer only the rewritten sentence.
        Example:
        If this was the preposition: "Options are: Minimalist, Trendy, or Surprise Me."
        Your answer should be something like this: "For the product subscription, customers can choose the style of jewelry pieces they want: Minimalist, Trendy, or Surprise Me."
        This is a good example because it includes the context that we are refering to the product subscription, we refer who has the option to choose, and we use words that are used on the rest of the text, like style instead of options"""
        response = openai_chat_completion(prompt, "Sentence: " + chunk + "\nText: " + text)
        return response

    def manual_chunks_filter(chunks, text):
        chunksToKeep = []
        for chunk in chunks:
            print("\nchunk: ", chunk, "\n")
            toKeep = input("""Do you want to keep this chunk as it is? if Yes type y
            If it is too summarized and needs context, type 1
            If you want to rewrite it yourself, type r""")
            if toKeep == "y":
                chunksToKeep.append(chunk)
            elif toKeep == "1":
                new_chunk = add_context(chunk, text)
                print("new chunk: ", new_chunk)
                toKeep = input("Write y to include like this, b to include it as before and r to rewrite it yourself")
                if toKeep == "y":
                    chunksToKeep.append(new_chunk)
                elif toKeep == "b":
                    chunksToKeep.append(chunk)
            elif toKeep == "r":
                new_chunk = input("Write the new chunk: ")
                chunksToKeep.append(new_chunk)
        return chunksToKeep

    """----------------------Private Methods-----------------------"""

    def _verify_qdrant_collection_structure(self):
        """
        Verify if the structure of the Qdrant collection matches the SCS collection.
        """
        pass
            

def main():
    collection = SCS_Collection.download_qdrant_collection("fruit_example", qdrant_tracker)
    collection.print()
    collection.append_scs(SCS("This is a new SCS from append_scs"))
    collection.append_sentence("This is a new sentence", "source_example")
    collection.print()
    collection.add_scs(1, SCS("This is a new SCS from add_scs"))
    collection.add_sentence(2, "This is a new sentence from add_sentence", "source_example_2")
    collection.print()
    print(f"Getting SCS at index 1: {collection.get_scs(1)}")
    print(f"Getting sentence at index 2: {collection.get_sentence(2)}")
    collection.delete_scs(1)
    collection.print()
    collection.delete_all_scs()
    collection.print()
    collection.append_scs(SCS("This is a new SCS after deletion"))
    collection.save(qdrant_tracker)
    collection.print()

if __name__ == '__main__':
    main()
