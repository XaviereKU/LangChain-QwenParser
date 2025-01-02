from typing import Optional as Optional
from typing import Union
import json
from langchain_core.output_parsers.transform import BaseTransformOutputParser

class QwenOutputParser(BaseTransformOutputParser[str]):
    """OutputParser that parses Qwen-2.5 Result"""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "output_parser"]

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "default"
    
    def parse(self, text) -> Union[str, dict]:
        """
        Returns the input text if no tool called.
        If tool callsed returns Dictionary
        """
        splited_text = list(map(lambda x: x.strip(), text.split('\n')))
        splited_text = list(filter(None, splited_text))
        
        if splited_text[0] == '<tool_call>':
            return json.loads(splited_text[1])
            
        elif '<tool_call>' in text:
            idx = splited_text.index('<tool_call>')
            return json.loads(splited_text[idx+1])
            
        else:
            return text

QwenOutputParser.model_rebuild()