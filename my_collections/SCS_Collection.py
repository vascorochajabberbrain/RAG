import QdrantTracker
from my_collections.Colletion import Collection
from objects.SCS import SCS


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
                    self.delete_scs(idx)
                case "p":
                    self.print()
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

    """----------------------Private Methods-----------------------"""

    def _verify_qdrant_collection_structure(self):
        """
        Verify if the structure of the Qdrant collection matches the SCS collection.
        """
        pass
            

def main():
    qdrant_tracker = QdrantTracker.QdrantTracker()
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
