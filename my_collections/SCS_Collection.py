from typing import final
import QdrantTracker
from qdrant_client.http.models import PointStruct
from my_collections.Colletion import Collection
from objects.SCS import SCS
from qdrant_utils import connect_to_upload_with_qdrant, insert_points
from vectorization import get_embedding, get_point_id, get_unique_id


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

    #outdated
    def append_scs(self, scs):
        """
        Append a new SCS to the collection.
        """
        if not isinstance(scs, SCS):
            raise TypeError("Expected an instance of SCS.")
        self.items.append(scs)

    def append_sentence(self, sentence, source=None):
        self.append_item(SCS(sentence, source))

    def append_sentences(self, sentences, source=None):
        if not isinstance(sentences, list):
            raise TypeError("Expected a list of sentences.")
        for sentence in sentences:
            if not isinstance(sentence, str):
                raise TypeError("Expected a string in the list of sentences.")
            self.append_item(SCS(sentence, source))

    #outdated
    def insert_scs(self, idx, scs):
        self._check_index(idx)
        if not isinstance(scs, SCS):
            raise TypeError("Expected an instance of SCS.")
        self.items.insert(idx, scs)

    def insert_sentence(self, idx, sentence, source=None):
        self.insert_item(idx, SCS(sentence, source))

    #outdated
    def get_scs(self, idx):
        self._check_index
        return self.items[idx]
    
    def get_sentence(self, idx):
        return self.get_item(idx).get_sentence()

    #outdated
    def delete_scs(self, idx):
        self._check_index(idx)
        del self.items[idx]

    #outdated
    def delete_all_scs(self):
        self.items = []

    """----------------------Private Methods-----------------------"""

    def _verify_qdrant_collection_structure(self):
        """
        Verify if the structure of the Qdrant collection matches the SCS collection.
        """
        pass

      


    #unused
    def _from_payload(payload):
        return SCS(payload["text"], payload["source"])

    #outdated
    def _to_embed(self, idx):
        """
        Convert the SCS at the given index to a string.
        """
        scs = self.items[idx]
        return scs.get_sentence()

    def _to_payload(self, idx):


        """
        Convert the SCS at the given index to a payload format suitable for Qdrant.
        """
        scs = self.items[idx]
        return {
            "text": scs.get_sentence(),
            "source": scs.get_source(),
            "idx": idx
        }
    
    #outdated
    # This method is not overwritable on children
    @final
    def make_qdrant_point(self, idx):
        return PointStruct(id=get_unique_id(), vector=get_embedding(self._to_embed(idx)), payload=self._to_payload(idx))
    
    def upload_qdrant_batch(self, batch):
        try:
            insert_points(self.collection_name, batch)
        except Exception as e:
            #to-do define the case when we had time out and try to make it each at a time
            print(f"Error inserting batch: {e}")
            for point in batch:
                try:
                    insert_points(self.collection_name, [point])
                except Exception as e_single:
                    print(f"Error inserting single point: {e_single}")

    def upload_qdrant_collection(self, batch_size=5):
        """
        Upload the SCS List to Qdrant.
        """
        # validate naming, only continue when we have a collection on Qdrant side to upload the SCSs
        connected_collection_name = connect_to_upload_with_qdrant(self.collection_name)
        if connected_collection_name != self.collection_name:
            print(f"SCS_Collection: Collection name updated to {connected_collection_name}.")
            self.collection_name = connected_collection_name
        
        batch = []
        for idx in range(len(self.items)):
            batch.append(self.make_qdrant_point(idx))

            if len(batch) == batch_size:
                self.upload_qdrant_batch(batch)
                batch = []  # Reset the batch after insertion

        if batch:
            self.upload_qdrant_batch(batch)

            

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
