'''Still supposing only one source per group.'''
from llms.openai_utils import openai_chat_completion

from objects.SCS import SCS
from objects.Item import Item


class Group(Item):
    LIMIT_OF_SCS = 8
    """-----------------------------Constructors-----------------------------"""
    
    def __init__(self, description=None, scss=None, source=None):
        super().__init__(source)
        self.description = description if description else ""
        self.scss = scss if scss else []

    @classmethod
    def from_payload(cls, payload):
        description = payload["description"]
        source = payload.get("source", None)
        self = cls(description, source=source)
        self.append_scss(self._string_to_scss(payload["text"]))
        return self
    
    """---------------------------------Dunders---------------------------------"""

    def __str__(self):
        text = f"Description: {self.description}\n"
        for i, prep in enumerate(self.scss):
            text += f"   ({i}) {prep}\n"
        if self.source:
            text += f"   Source: {self.source}\n"
        return text
    
    def __repr__(self):
        if self.source:
            return f"Group(description={self.description}, scss={self.scss}, source={self.source})"
        else:
            return f"Group(description={self.description}, scss={self.scss})"
    
    """-----------------------------Public Methods-----------------------------"""
    """--------Status related methods---------"""

    def is_full(self):
        """
        Check if the group is full.
        """
        return len(self.scss) >= Group.LIMIT_OF_SCS
    
    def is_empty(self):
        return len(self.scss) == 0
    
    """----------SCS related methods----------"""

    #outdated I think
    def append_sentence(self, scss):# one or more
        """
        Add SCS's to the group.
        """
        self._check_full()
        if isinstance(scss, str):
            self.scss.append(scss)
        elif isinstance(scss, list):
            for i, scs in enumerate(scss):
                if not isinstance(scs, str):
                    raise TypeError(f"Expected a string, got {type(scs)} at index {i}.")
            self.scss.extend(scss)
        else:
            raise TypeError("Expected a string or a list of strings for SCS.")

    def append_scs(self, scs):
        if isinstance(scs, str):
            scs = SCS(scs)
        self._check_full()
        if not isinstance(scs, SCS):
            raise TypeError("Expected a SCS object.")
        self.append_sentence(scs.get_sentence())

    def append_scss(self, scss):
        """
        Add multiple SCS objects to the group.
        """
        for scs in scss:
            self.append_scs(scs)

    def get_scs(self, idx):#only one
        """
        Get a specific SCS by index.
        """
        self._check_index(idx)
        return self.scss[idx]
        
    def get_all_scs(self):
        """
        Get all SCS's in the group.
        """
        return self.scss
    
    def delete_scs(self, idxs):#one or more
        """
        Delete SCS's by index.
        """
        if isinstance(idxs, int):
            self._check_index(idxs)
            del self.scss[idxs]
        elif isinstance(idxs, range):
            idxs = list(idxs)
        elif isinstance(idxs, list):
            for i, idx in enumerate(idxs):
                if not isinstance(i, int):
                    raise TypeError(f"Expected an integer, got {type(idx)} at index {i}.")
                self._check_index(idx)
            idxs.sort(reverse=True)  # Sort in reverse order to avoid index shifting
            for idx in idxs:
                del self.scss[idx]
        else:
            raise TypeError("Expected an integer or a list of integers for SCS index.")


    """--------Description related methods--------"""

    def set_description(self, new_description):
        """
        Set a new description for the group.
        """
        if not isinstance(new_description, str):
            raise TypeError(f"Expected a string for description, got {type(new_description)}.")
        if not new_description:
            raise ValueError("Description cannot be an empty string.")
        self.description = new_description

    def get_description(self):
        """
        Get the description of the group.
        """
        return self.description
    
    def delete_description(self):
        """
        Delete the description from the group.
        """
        self.description = None
    
    def create_description(self):
        """
        Use an LLM to create a description for the group based on its SCS's.
        """
        prompt = f"""You are an expert in creating concise and accurate descriptions for groups of sentences.
                Given a list of sentences, your task is to generate a brief description that encapsulates the main theme or subject of the sentences.
                The description should be no longer than 15 words and should be clear and to the point.

                The user will send you a list of sentences like this:
                "A dog's sense of smell is remarkably powerful, estimated to be anywhere from 10,000 to 100,000 times more acute than a human's. This allows them to detect minute scents, which is why they're used in search-and-rescue, bomb detection, and medical diagnosis.
                Dogs can be trained to detect diseases. Some can be trained to sniff out diseases like cancer, diabetes, and even COVID-19 by identifying specific odors released by the human body.
                The average dog is as intelligent as a two-year-old human. They can understand over 150 words and gestures and are capable of counting up to four or five.
                The Basenji breed is unique because it doesn't bark. Instead, it makes a yodel-like sound due to its unusually shaped larynx."

                Which a description for these sentences would be:
                "Facts about dogs' abilities and intelligence."
                
                The format of your response must be only the actual description, without any additional explanation or punctuation."""
        text = self._scss_to_string()
        llm_response = openai_chat_completion(prompt, text)
        llm_response = self._check_valid_llm_response(llm_response)
        self.set_description(llm_response)
        return llm_response
    """---------Qdrant related methods-----------"""

    def to_payload(self, index=None):
        if index is not None and self.source is not None:
            return {
                "description": self.description,
                "text": self._scss_to_string(),
                "source": self.source,
                "idx": index
            }
        if self.source is not None:
            return {
                "description": self.description,
                "text": self._scss_to_string(),
                "source": self.source
            }
        if index is not None:
            return {
                "description": self.description,
                "text": self._scss_to_string(),
                "idx": index
            }
        else:
            return {
                "description": self.description,
                "text": self._scss_to_string()
            }
        
    def to_embed(self):
        """
        Convert the group to a string suitable for embedding.
        """
        return self.description + self._scss_to_string()
    
    """-----------------------------Private Methods-----------------------------"""

    def _scss_to_string(self):
        return "\n".join(self.scss)
    
    def _string_to_scss(self, string):
            if string == "":
                return []
            return string.split("\n")

    def _check_index(self, idx):
        """
        Check if the index is not valid for the current group.
        """
        if idx < 0 or idx >= len(self.scss):
            raise IndexError("Index out of bounds for this Group.")
        
    def _check_full(self):
        """
        Check if the group is full.
        """
        if self.is_full():
            raise ValueError("Group is full. Cannot add more SCS's.")
        
    #not very useful for now but I like the principle of allways processing the llm response before using it
    def _check_valid_llm_response(self, response):
        """
        Check if the LLM response is valid.
        """
        if not response:
            raise ValueError("LLM response is empty.")
        return response
        