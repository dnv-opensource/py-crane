"""Module introducing the Sphinx extension get-from-code, fetching docstrings, string representations, ... from code."""

import importlib
import logging

from docutils import nodes  # type: ignore

# from docutils.parsers.rst import Directive
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)


class GetFromCode(SphinxDirective):
    # this enables content in the directive
    required_arguments = 1
    optional_arguments = 1
    has_content = True

    def run(self):
        """Extract the value of the (constant) variable or the docstring according to spec."""
        if len(self.arguments) == 1:
            self.typ = None
        else:
            self.typ = self.arguments[1]
        sPkg, _, sObj = self.arguments[0].partition(".")
        sMod, _, sObj = sObj.partition(".")
        obj = importlib.import_module(sPkg + "." + sMod)
        while len(sObj):
            sSub, _, sObj = sObj.partition(".")
            obj = getattr(obj, sSub)
            assert obj is not None, f"Attribute {sSub} of {obj} not found"
        logger.debug(
            "INFO [" + self.arguments[0] + "] Type:",
            type(obj),
            ", __name__: " + type(obj).__name__,
            ", type: ",
            self.typ,
        )
        if self.typ is not None:
            if self.typ == "code":
                text = (
                    obj.__code__
                )  # TODO: this is only the code object. Need to retrieve the code text and format it appropriately
            else:
                text = "Type " + self.typ + " not yet implemented"
        elif isinstance(obj, (complex, float, int, str, list, dict, tuple)):  # no docstring
            text = str(obj).strip()
        elif isinstance(obj.__doc__, str):
            text = obj.__doc__.strip()
        else:
            raise NotImplementedError(f"Unhandled situation obj:{obj}") from None

        if self.typ is None:  # need to parse the extracted text
            parNode = nodes.paragraph(text="")  # use an empty paragraph node as parent
            self.state.nested_parse(self.content, 0, parNode)  # here the content of the retrieved text is parsed
        else:  # use the text as is
            parNode = nodes.paragraph(text=text)
        return [parNode]


def setup(app):
    app.add_directive("get-from-code", GetFromCode)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
