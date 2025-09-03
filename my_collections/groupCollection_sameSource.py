from types import NoneType
from my_collections.groupCollection import GroupCollection
from objects.SCS import SCS


class groupCollection_sameSource(GroupCollection):
    TYPE = "group_sameSource"
    LIMIT_OF_UNFULL_GROUPS = 5

    def unfull_groups_indexes(self, scs):
        source = scs.get_source()
        indexes = []
        for idx, group in enumerate(self.items):
            if not group.is_full() and group.same_source(source):
                indexes.append(idx)
        return indexes
        
    def collection_is_full(self, scs):        
        return len(self.unfull_groups_indexes(scs)) >= groupCollection_sameSource.LIMIT_OF_UNFULL_GROUPS
    
    def append_description(self, description, source):

        return super().append_description(description, source)
    
    def append_scs(self, group_idx, scs):
        if isinstance(scs, str):
            scs = SCS(scs)
        if not isinstance(scs, SCS):
            raise TypeError("Expected a SCS object.")
        group = self.items[group_idx]
        if group.is_empty():
            group.set_source(scs.get_source())
        elif not group.same_source(scs.get_source()):
            raise ValueError("SCS must have the same source as the group.")
        super().append_scs(group_idx, scs)