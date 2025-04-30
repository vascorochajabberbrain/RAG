from qdrant_client.http.models import PointStruct

from vectorization import get_embedding, get_point_id


class GroupCollection:
    def __init__(self):
        self.groups = []

    def add_group(self, description):
        self.groups.append({
            "description": description,
            "prepositions": []
        })

    def add_groups(self, descriptions):
        for description in descriptions:
            self.add_group(description)

    def add_preposition(self, group_index, preposition):
        self._validate_group_index(group_index)
        self.groups[group_index]["prepositions"].append(preposition)

    def get_prepositions(self, group_index):
        self._validate_group_index(group_index)
        return self.groups[group_index]["prepositions"]

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

    def from_save_points(self, points):
        for point in points:
            self._payload_to_group(point.payload)

    def print(self):
        for i, group in enumerate(self.groups):
            print(f"[{i}] {group['description']}")
            for j, prep in enumerate(group["prepositions"]):
                print(f"   ({j}) {prep}")
            print()

    def to_string(self):
        lines = []
        for i, group in enumerate(self.groups):
            lines.append(f"[{i}] {group['description']}")
            for j, prep in enumerate(group["prepositions"]):
                lines.append(f"   ({j}) {prep}")
            lines.append("")  # Blank line between groups
        return "\n".join(lines)

    # Placeholder for future search implementation
    def search_prepositions(self, group_index, keyword):
        pass

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
