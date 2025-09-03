from my_collections.Colletion import Collection
from objects.Group import Group


class GroupCollection(Collection):
    TYPE = "group"
    LIMIT_OF_UNFULL_GROUPS = 12

    """------------------------------Constructors------------------------------"""

    def init_item_from_qdrant(self, point_data):
        return Group.from_payload(point_data)

    """-----------------------------Public Methods-----------------------------"""

    #outdated now
    def print_only_not_full_groups(self):
        not_full_groups = [idx for idx, group in enumerate(self.items) if not group.is_full()]
        self.print(not_full_groups)

    def collection_is_full(self, scs):
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

    def append_description(self, description, scs):
        """
        Append a new Group with the given description.
        """
        if not isinstance(description, str):
            raise TypeError("Expected a string for description.")
        self._check_full_collection(scs)
        self.append_item(Group(description))
        return len(self.items) - 1  # Return the index of the newly added group

    def append_descriptions(self, descriptions):
        """
        Append multiple Groups with the given descriptions.
        """
        if not isinstance(descriptions, list):
            raise TypeError("Expected a list for descriptions.")
        for description in descriptions:
            self.append_description(description)

    def insert_description(self, idx, description):
        self._check_index(idx)
        if not isinstance(description, str):
            raise TypeError("Expected a string for description.")
        self.insert_item(idx, Group(description))    

    def available_groups_for_source(self, source):
        """
        Find a group by its source.
        """
        if not isinstance(source, str) and source is not None:
            raise TypeError("Expected a string for source or None.")
        indexes = []
        for idx, group in enumerate(self.items):
            if group.same_source(source) and not group.is_full():
                indexes.append(idx)
        return indexes
    
    def unfull_groups_indexes(self, scs):
        return [idx for idx, group in enumerate(self.items) if not group.is_full()]
    
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
    
    def _check_full_collection(self, scs):
        if self.collection_is_full(scs):
            raise Exception("The collection is full, cannot add more groups before fulling another group.")

    
    #only exists for grouped_VDB.py
    def existing_group_index(self, index):
        return 0 <= index < len(self.items)
    

def main():
    """The main now serves as a test for the GroupCollection class."""
    try:
        collection = GroupCollection()
        collection.print()
        collection.append_item(Group("This is a new group from append_item"))
        collection.append_description("This is a new description from append_description")
        collection.append_descriptions(["Description 1", "Description 2"])
        collection.print()
        collection.insert_item(1, Group("This is a new group from insert_item"))
        collection.insert_description(2, "This is a new group from insert_description")
        collection.print()
        collection.delete_item(1)
        
        collection.print()
        collection.append_scs(1, "This is a new SCS in group 1")
        collection.append_scs(1, "This is second SCS in group 1")
        collection.append_scs(1, "This is third SCS in group 1")
        collection.append_scs(1, "This is another new SCS in group 1")
        collection.append_scs(1, "This is a fifth new SCS in 1")
        collection.append_scs(1, "This is a sixth new SCS in 1")
        collection.append_scs(1, "This is a seventh new SCS in 1")
        collection.append_scs(1, "This is a eighth new SCS in 1")
        print("--------------------now, first just print")
        collection.print()
        print("--------------------now, print only full groups")
        collection.print_only_not_full_groups()
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

if __name__ == '__main__':
    main()