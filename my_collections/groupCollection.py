from qdrant_client.http.models import PointStruct


from objects.Group import Group
import QdrantTracker
from vectorization import get_embedding, get_point_id


class GroupCollection:
    TYPE = "group"
    LIMIT_OF_UNFULL_GROUPS = 12
    """------------------------------Constructors------------------------------"""

    def init_item_from_qdrant(self, point_data):
        return Group.from_payload(point_data)

    """-----------------------------Public Methods-----------------------------"""

    def collection_is_full(self):
        #a collection is full when the number of unfull groups reaches the limit
        return self._number_of_groups() - self._number_of_full_groups() >= GroupCollection.LIMIT_OF_UNFULL_GROUPS
    
    def menu(self):
        menu = """Select an action:
        -- "q" to quit
        -- "app" to append a group
        -- "ins" to insert a group (by index)
        -- "del" to delete a group
        -- "app_scs" to append a scs to a group
        -- "del_scs" to delete a scs from a group
        -- "m_scs" to move a scs from a group to another
        -- "c_scs" to copy a scs from a group to another
        -- "u" to update a group description
        -- "p" to print the collection
        -- "s" to save the collection
"""
        action = input(menu)

        while action != "q":
            match action:
                case "app":
                    description = input("Description:")
                    self.append_description(description)
                case "ins":
                    idx = int(input("Group Index:"))
                    description = input("Description:")
                    self.insert_description(idx, description)
                case "del":
                    idx = int(input("Group Index:"))
                    self.delete_group(idx)
                case "app_scs":
                    group_idx = int(input("Group Index:"))
                    scs = input("SCS:")
                    self.append_scs(group_idx, scs)
                case "del_scs":
                    group_idx = int(input("Group Index:"))
                    scs_idx = int(input("SCS Index:"))
                    self.delete_scs(group_idx, scs_idx)
                case "m_scs":
                    group_idx_from = int(input("Group Index From:"))
                    group_idx_to = int(input("Group Index To:"))
                    scs_idx = int(input("SCS Index:"))
                    self.move_scs(group_idx_from, group_idx_to, scs_idx)
                case "c_scs":
                    group_idx_from = int(input("Group Index From:"))
                    group_idx_to = int(input("Group Index To:"))
                    scs_idx = int(input("SCS Index:"))
                    self.copy_scs(group_idx_from, group_idx_to, scs_idx)
                case "u":
                    group_idx = int(input("Group Index:"))
                    description = input("Description:")
                    self.update_description(group_idx, description)
                case "p":
                    self.print()
                case "s":
                    return True
                case _:
                    print("Invalid action.")
            action = input(menu)
        return False
    
    """----------Group related methods----------"""
    #outdated
    def append_group(self, group):
        """
        Append a new Group to the collection.
        """
        if not isinstance(group, Group):
            raise TypeError("Expected an instance of Group.")
        self.items.append(group)
    #outdated
    def append_groups(self, groups):
        for group in groups:
            self.append_group(group)

    def append_description(self, description):
        """
        Append a new Group with the given description.
        """
        if not isinstance(description, str):
            raise TypeError("Expected a string for description.")
        self.append_item(Group(description))

    def append_descriptions(self, descriptions):
        """
        Append multiple Groups with the given descriptions.
        """
        if not isinstance(descriptions, list):
            raise TypeError("Expected a list for descriptions.")
        for description in descriptions:
            self.append_description(description)

    #outdated
    def insert_group(self, idx, group):
        self._check_index(idx)
        if not isinstance(group, Group):
            raise TypeError("Expected an instance of Group.")
        self.items.insert(idx, group)

    def insert_description(self, idx, description):
        self._check_index(idx)
        if not isinstance(description, str):
            raise TypeError("Expected a string for description.")
        self.insert_item(idx, Group(description))

    #outdated
    def delete_group(self, group_index):
        self._check_index(group_index)
        del self.items[group_index]
    

    """-----------SCS related methods-----------"""
    def append_scs(self, group_index, scs):
        self._check_index(group_index)
        self.items[group_index].append_scs(scs)

    def get_scss(self, group_index):
        self._check_index(group_index)
        return self.items[group_index].get_all_scs()
    
    # Placeholder for future search implementation
    def search_scss(self, group_index, keyword):
        pass

    #this can actually receive a list on the scs_index
    def delete_scs(self, group_index, scs_index):
        self._check_index(group_index)
        self.items[group_index].delete_scs(scs_index)

    def move_scs(self, from_group_index, to_group_index, scs_index):
        self._check_index(from_group_index)
        self._check_index(to_group_index)
        scs = self.items[from_group_index].get_scs(scs_index)
        self.items[to_group_index].add_scs(scs)
        self.items[from_group_index].delete_scs(scs_index)

    def copy_scs(self, from_group_index, to_group_index, scs_index):
        self._check_index(from_group_index)
        self._check_index(to_group_index)
        scs = self.items[from_group_index].get_scs(scs_index)
        self.items[to_group_index].append_scs(scs)

    """--------Description related methods--------"""
    def update_description(self, group_index, new_description):
        self._check_index(group_index)
        self.items[group_index].set_description(new_description)

    def get_description(self, group_index):
        self._check_index(group_index)
        return self.items[group_index].get_description()
    
    def get_all_descriptions(self):
        return [group.get_description() for group in self.items]
    
    """------------Private methods--------------"""
    
    def _number_of_groups(self):
        return len(self.items)
    
    def _number_of_full_groups(self):
        return sum(1 for group in self.items if group.is_full())
    
    def _group_is_full(self, group_index):
        self._check_index(group_index)
        return self.items[group_index].is_full()
    
    #outdated
    def to_string(self):
        lines = []
        for i, group in enumerate(self.items):
            if self._group_is_full(i):
                continue
            lines.append(f"[{i}] {group['description']}")
            for j, prep in enumerate(group["prepositions"]):
                lines.append(f"   ({j}) {prep}")
            lines.append("")  # Blank line between groups
        return "\n".join(lines)

    #outdated
    def to_save_points(self):
        points = []
        for idx, group in enumerate(self.items):
            to_embed = group["description"] + "\n\n" + self._scss_to_string(idx)
            
            points.append(PointStruct(id=get_point_id(), vector=get_embedding(to_embed), payload=self._group_to_payload(idx)))
        return points
    #outdated
    def to_save_points_w_batch(self, batch_size=5):
        batch = []
        for idx, group in enumerate(self.items):
            to_embed = group["description"] + "\n\n" + self._scss_to_string(idx)
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
    #outdated
    def from_save_points(self, points):
        for point in points:
            self._payload_to_group(point.payload)

    
    #only exists for grouped_VDB.py
    def existing_group_index(self, index):
        return 0 <= index < len(self.items)
    
    #outdated
    def _scss_to_string(self, group_index):
        scss = self.items[group_index]["prepositions"]
        return "\n".join(scss)
    #outdated
    def _string_to_scss(self, string):
        return string.split("\n")
    #outdated
    def _group_to_payload(self, group_index):
        return {
            "description": self.items[group_index]["description"],
            "text": self._scss_to_string(group_index)
        }
    #outdated
    def _payload_to_group(self, payload):
        self.items.append({
            "description": payload["description"],
            "prepositions": self._string_to_scss(payload["text"])
        })

def main():
    qdrant_tracker = QdrantTracker.QdrantTracker()
    collection = GroupCollection.download_qdrant_collection("testing_w_groups", qdrant_tracker)
    try:
        """
        collection.print()
        collection.append_group(Group("This is a new group from append_group"))
        collection.append_groups([Group("Group 1"), Group("Group 2")])
        collection.append_description("This is a new description from append_description")
        collection.append_descriptions(["Description 1", "Description 2"])
        collection.print()
        collection.add_group(1, Group("This is a new group from add_group"))
        collection.add_description(2, "This is a new group from add_description")
        collection.print()
        collection.delete_group(1)
        """
        collection.print()
        print(f"going to add")
        collection.add_scs(1, "This is a new SCS in group 1")
        collection.add_scs(1, "This is a new SCS in group 1")
        collection.add_scs(1, "This is a new SCS in group 1")
        collection.add_scs(1, "This is another new SCS in group 1")
        collection.add_scs(1, "This is a third new SCS in group 1")
        collection.add_scs(1, "This is a fourth new SCS in 1")
        collection.add_scs(1, "This is a fifth new SCS in 1")
        collection.add_scs(1, "This is a sixth new SCS in 1")
        collection.add_scs(1, "This is a seventh new SCS in 1")
        collection.print()
        print(collection.get_scss(1))
        print("going to delete 1, 0")
        collection.delete_scs(1, 0)
        collection.print()
        print("going to move 0, 0, 1")
        collection.move_scs(0, 0, 1)
        collection.print()
        print("going to copy 0, 0, 2")
        collection.copy_scs(0, 0, 2)
        collection.print()
        print("going to update description of group 0")
        collection.update_description(0, "Updated description for group 0")
        print(collection.get_description(0))
        print(collection.get_all_descriptions())
    except Exception as e:
        print(e)
    collection.save(qdrant_tracker) 

if __name__ == '__main__':
    main()