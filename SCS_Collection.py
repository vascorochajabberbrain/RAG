from typing import final
import QdrantTracker
from qdrant_client.http.models import PointStruct
from SCS import SCS
from qdrant_utils import connect_to_upload_with_qdrant, insert_points
from vectorization import get_embedding, get_point_id, get_unique_id


class SCS_Collection:
    """
    A class to represent a collection of self-contained sentences (SCS).
    """

    def __init__(self, collection_name=None):
        self.scs_list = []
        self.collection_name = collection_name

    def append_scs(self, scs):
        """
        Append a new SCS to the collection.
        """
        if not isinstance(scs, SCS):
            raise TypeError("Expected an instance of SCS.")
        self.scs_list.append(scs)

    def append_sentence(self, sentence, source=None):
        self.scs_list.append(SCS(sentence, source))

    def add_scs(self, idx, scs, source=None):
        self._validate_idx(idx)
        self.scs_list.insert(idx, scs)

    def get_scs(self, idx):
        self._validate_idx
        return self.scs_list[idx]

    def delete_scs(self, idx):
        self._validate_idx(idx)
        del self.scs_list[idx]

    def _validate_idx(self, idx):
        """
        Validate the index for accessing the SCS collection.
        """
        if idx < 0 or idx >= len(self.scs_list):
            raise IndexError("Index out of bounds for SCS collection.")

    def __str__(self):
        pass

    #this is a constructor also, just initializes with what we have on Qdrant
    @classmethod
    def download_qdrant_collection(cls, collection_name, qdrant_tracker: QdrantTracker):
        """
        Download the SCS List from Qdrant.
        """
        collection_name, qdrant_points = qdrant_tracker.connect(collection_name)
        #TO-DO: verify if _from_payload works on this collection type, might exist a missmatch
        self = cls(collection_name)
        print(f"Downloading collection: {collection_name}")
        for qdrant_point in qdrant_points:
            print(qdrant_point)
            self.append_scs(SCS.from_payload(qdrant_point))
        
        return self
        
    def save(self, qdrant_tracker: QdrantTracker):
        """
        Save the SCS Collection on Qdrant.
        """
        qdrant_points = []
        for scs in self.scs_list:
            print(scs)
            qdrant_points.append(PointStruct(
                id=get_point_id(),
                vector=get_embedding(scs.get_sentence()),  #maybe on a broad version of collection it should be scs.to_embed()
                payload=scs.to_payload()))
        qdrant_tracker.disconnect(self.collection_name, qdrant_points)


    def print(self, list_indexes=None):
        if list_indexes is None:
            list_indexes = range(len(self.scs_list))
        if not (isinstance(list_indexes, list) or isinstance(list_indexes, range)) and len(self.scs_list) != 0:
            raise TypeError("list_indexes must be a list of integers")
        for i, scs in enumerate(self.scs_list):
            if i not in list_indexes:
                continue
            print(f"[{i}] {scs}")
            


    def _verify_qdrant_collection_structure(self):
        """
        Verify if the structure of the Qdrant collection matches the SCS collection.
        """
        pass


    def _to_embed(self, idx):
        """
        Convert the SCS at the given index to a string.
        """
        scs = self.scs_list[idx]
        return scs.get_sentence()

    
    def _to_payload(self, idx):


        """
        Convert the SCS at the given index to a payload format suitable for Qdrant.
        """
        scs = self.scs_list[idx]
        return {
            "text": scs.get_sentence(),
            "source": scs.get_source(),
            "idx": idx
        }
    
    #unused
    def _from_payload(payload):
        return SCS(payload["text"], payload["source"])

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
        for idx in range(len(self.scs_list)):
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
    collection.append_scs(SCS("This is a new SCS"))
    collection.print()
    collection.save(qdrant_tracker)
    collection.print()

if __name__ == '__main__':
    main()
