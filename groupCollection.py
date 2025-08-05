from qdrant_client.http.models import PointStruct


from Group import Group
import QdrantTracker
from vectorization import get_embedding, get_point_id


class GroupCollection:
    LIMIT_OF_SCS = 8
    LIMIT_OF_UNFULL_GROUPS = 12

    def __init__(self, collection_name=None):
        self.groups = []
        self.collection_name = collection_name

    def add_group(self, description):
        self.groups.append({
            "description": description,
            "prepositions": []
        })


    def add_groups(self, descriptions):
        for description in descriptions:
            self.add_group(description)

    def append_group(self, group):
        """
        Append a new Group to the collection.
        """
        if not isinstance(group, Group):
            raise TypeError("Expected an instance of Group.")
        self.groups.append(group)

    def add_preposition(self, group_index, preposition):
        self._validate_group_index(group_index)
        self.groups[group_index]["prepositions"].append(preposition)

    def get_prepositions(self, group_index):
        self._validate_group_index(group_index)
        return self.groups[group_index]["prepositions"]
    
    def number_of_groups(self):
        return len(self.groups)
    
    def number_of_full_groups(self):
        return sum(1 for group in self.groups if self.group_is_full(self.groups.index(group)))
    
    def number_of_prepositions(self, group_index):
        self._validate_group_index(group_index)
        return len(self.groups[group_index]["prepositions"])

    def move_preposition(self, preposition_index, from_group_index, to_group_index):
        self._validate_group_index(from_group_index)
        self._validate_group_index(to_group_index)
        preps = self.groups[from_group_index]["prepositions"]

        if preposition_index < 0 or preposition_index >= len(preps):
            raise IndexError("Invalid preposition index")

        prep = preps.pop(preposition_index)
        self.groups[to_group_index]["prepositions"].append(prep)

    def copy_preposition(self, preposition_index, from_group_index, to_group_index):
        self._validate_group_index(from_group_index)
        self._validate_group_index(to_group_index)
        preps = self.groups[from_group_index]["prepositions"]

        if preposition_index < 0 or preposition_index >= len(preps):
            raise IndexError("Invalid preposition index")

        prep = preps[preposition_index]
        self.groups[to_group_index]["prepositions"].append(prep)

    def update_description(self, group_index, new_description):
        self._validate_group_index(group_index)
        self.groups[group_index]["description"] = new_description

    def get_description(self, group_index):
        self._validate_group_index(group_index)
        return self.groups[group_index]["description"]
    def get_all_descriptions(self):
        return [group["description"] for group in self.groups]

    def delete_group(self, group_index):
        self._validate_group_index(group_index)
        del self.groups[group_index]

    def delete_preposition(self, group_index, preposition_index):
        self._validate_group_index(group_index)
        preps = self.groups[group_index]["prepositions"]
        if preposition_index < 0 or preposition_index >= len(preps):
            raise IndexError("Invalid preposition index")
        del preps[preposition_index]

    def to_save_points(self):
        points = []
        for idx, group in enumerate(self.groups):
            to_embed = group["description"] + "\n\n" + self._prepositions_to_string(idx)
            
            points.append(PointStruct(id=get_point_id(), vector=get_embedding(to_embed), payload=self._group_to_payload(idx)))
        return points
    
    def to_save_points_w_batch(self, batch_size=5):
        batch = []
        for idx, group in enumerate(self.groups):
            to_embed = group["description"] + "\n\n" + self._prepositions_to_string(idx)
            point = PointStruct(
                id=get_point_id(),
                vector=get_embedding(to_embed),
                payload=self._group_to_payload(idx)
            )
            batch.append(point)

            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch  # Yield any remaining points at the end

    def from_save_points(self, points):
        for point in points:
            self._payload_to_group(point.payload)

    def print(self, list_indexes=None):
        if list_indexes is None:
            list_indexes = range(len(self.groups))
        if not (isinstance(list_indexes, list) or isinstance(list_indexes, range))  and len(self.groups) != 0:
            raise TypeError("list_indexes must be a list of integers")
        for i, group in enumerate(self.groups):
            if i not in list_indexes:
                continue
            print(f"[{i}] {group}")

    def to_string(self):
        lines = []
        for i, group in enumerate(self.groups):
            if self.group_is_full(i):
                continue
            lines.append(f"[{i}] {group['description']}")
            for j, prep in enumerate(group["prepositions"]):
                lines.append(f"   ({j}) {prep}")
            lines.append("")  # Blank line between groups
        return "\n".join(lines)

    @classmethod
    def download_qdrant_collection(cls, collection_name, qdrant_tracker: QdrantTracker):
        """
        Download the SCS List from Qdrant.
        """
        
        collection_name, qdrant_points = qdrant_tracker.connect(collection_name)
        #TO-DO: verify if _from_payload works on this collection type, might exist a missmatch
        self = cls(collection_name)
        print(f"Downloading collection: {collection_name} ({len(qdrant_points)} points)")
        for qdrant_point in qdrant_points:
            print(qdrant_point)
            self.append_group(Group.from_payload(qdrant_point))
        
        return self


    def save(self, qdrant_tracker: QdrantTracker):
        """
        Save the Group Collection on Qdrant.
        """
        qdrant_points = []
        for group in self.groups:
            print(group)
            qdrant_points.append(PointStruct(
                id=get_point_id(),
                vector=get_embedding(group.to_embed()),
                payload=group.to_payload()
            ))
        
        qdrant_tracker.disconnect(self.collection_name, qdrant_points)

    # Placeholder for future search implementation
    def search_prepositions(self, group_index, keyword):
        pass

    def group_is_full(self, group_index):
        self._validate_group_index(group_index)
        return self.number_of_prepositions(group_index) >= self.LIMIT_OF_SCS
    
    def collection_is_full(self):
        #a collection is full when the number of unfull groups reaches the limit
        return self.number_of_groups() - self.number_of_full_groups() >= self.LIMIT_OF_UNFULL_GROUPS
    
    def existing_group_index(self, index):
        return 0 <= index < len(self.groups)
    #these two are very similar for now...
    def _validate_group_index(self, index):
        if index < 0 or index >= len(self.groups):
            raise IndexError("Invalid group index")
        
    def _prepositions_to_string(self, group_index):
        prepositions = self.groups[group_index]["prepositions"]
        return "\n".join(prepositions)
    
    def _string_to_prepositions(self, string):
        return string.split("\n")
    
    def _group_to_payload(self, group_index):
        return {
            "description": self.groups[group_index]["description"],
            "text": self._prepositions_to_string(group_index)
        }
    
    def _payload_to_group(self, payload):
        self.groups.append({
            "description": payload["description"],
            "prepositions": self._string_to_prepositions(payload["text"])
        })

def main():
    qdrant_tracker = QdrantTracker.QdrantTracker()
    collection = GroupCollection.download_qdrant_collection("testing_w_groups", qdrant_tracker)
    collection.print()
    collection.append_group(Group("This is a new group"))
    collection.append_group(Group("This is a new preposition", ["This is a new preposition", "This is another new preposition"]))
    collection.print()
    collection.save(qdrant_tracker) 
    collection.print()

if __name__ == '__main__':
    main()