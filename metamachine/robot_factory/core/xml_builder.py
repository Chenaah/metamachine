"""
Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Dict, List
from lxml import etree


class XMLBuilderError(Exception):
    """Base exception class for XMLBuilder errors."""
    pass

class XMLBuilder:
    """Helper class for building XML elements with proper validation."""
    
    @staticmethod
    def create_element(parent: etree.Element, tag: str, **attrs) -> etree.Element:
        """Create an XML element with validation."""
        try:
            if parent is None:
                return etree.Element(tag, **attrs)
            return etree.SubElement(parent, tag, **attrs)
        except Exception as e:
            raise XMLBuilderError(f"Failed to create {tag} element: {str(e)}")

    @staticmethod
    def validate_attributes(attrs: Dict[str, str], required: List[str]) -> None:
        """Validate that required attributes are present."""
        missing = [attr for attr in required if attr not in attrs]
        if missing:
            raise XMLBuilderError(f"Missing required attributes: {', '.join(missing)}")
