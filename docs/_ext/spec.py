"""Extension for 'spec-type' Sphinx directives
`Build along the lines of the tutorial <https://www.sphinx-doc.org/en/master/development/tutorials/todo.html>`_
With base extension 'spec' and sub-classes 'assumption', 'limitation' and 'requirement'. These just change the title of the admonition.
"""

from docutils import nodes  # type: ignore
from docutils.parsers.rst import Directive  # type: ignore
from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective


class Spec(nodes.Admonition, nodes.Element):
    pass


class SpecList(nodes.General, nodes.Element):
    pass


def visit_spec_node(self, node):
    self.visit_admonition(node)


def depart_spec_node(self, node):
    self.depart_admonition(node)


class SpeclistDirective(Directive):
    def run(self):
        return [SpecList("")]


class SpecDirective(SphinxDirective):
    _title = "Spec"
    # this enables content in the directive
    has_content = True

    def run(self):
        targetid = "spec-%d" % self.env.new_serialno("spec")
        targetnode = nodes.target("", "", ids=[targetid])

        spec_node = Spec("\n".join(self.content))
        spec_node += nodes.title(_("Spec"), _(self._title))
        self.state.nested_parse(self.content, self.content_offset, spec_node)

        if not hasattr(self.env, "spec_all_specs"):
            self.env.spec_all_specs = []  # type: ignore[attr-defined]

        self.env.spec_all_specs.append(  # type: ignore[attr-defined]
            {
                "docname": self.env.docname,
                "lineno": self.lineno,
                "spec": spec_node.deepcopy(),
                "target": targetnode,
            }
        )

        return [targetnode, spec_node]


class LimitationDirective(SpecDirective):
    _title = "Limitation"


class AssumptionDirective(SpecDirective):
    _title = "Assumption"


class RequirementDirective(SpecDirective):
    _title = "Requirement"


def purge_specs(app, env, docname):
    if not hasattr(env, "spec_all_specs"):
        return

    env.spec_all_specs = [spec for spec in env.spec_all_specs if spec["docname"] != docname]


def merge_specs(app, env, docnames, other):
    if not hasattr(env, "spec_all_specs"):
        env.spec_all_specs = []
    if hasattr(other, "spec_all_specs"):
        env.spec_all_specs.extend(other.spec_all_specs)


def process_spec_nodes(app, doctree, fromdocname):
    if not app.config.spec_include_specs:
        for node in doctree.findall(Spec):
            node.parent.remove(node)

    # Replace all speclist nodes with a list of the collected specs.
    # Augment each spec with a backlink to the original location.
    env = app.builder.env

    if not hasattr(env, "spec_all_specs"):
        env.spec_all_specs = []

    for node in doctree.findall(SpecList):
        if not app.config.spec_include_specs:
            node.replace_self([])
            continue

        content = []

        for spec_info in env.spec_all_specs:
            para = nodes.paragraph()
            filename = env.doc2path(spec_info["docname"], base=None)
            description = _("(The original entry is located in %s, line %d and can be found ") % (
                filename,
                spec_info["lineno"],
            )
            para += nodes.Text(description)

            # Create a reference
            newnode = nodes.reference("", "")
            innernode = nodes.emphasis(_("here"), _("here"))
            newnode["refdocname"] = spec_info["docname"]
            newnode["refuri"] = app.builder.get_relative_uri(fromdocname, spec_info["docname"])
            newnode["refuri"] += "#" + spec_info["target"]["refid"]
            newnode.append(innernode)
            para += newnode
            para += nodes.Text(".)")

            # Insert into the speclist
            content.append(spec_info["spec"])
            content.append(para)

        node.replace_self(content)


def setup(app):
    app.add_config_value("spec_include_specs", False, "html")

    app.add_node(SpecList)
    app.add_node(
        Spec,
        html=(visit_spec_node, depart_spec_node),
        latex=(visit_spec_node, depart_spec_node),
        text=(visit_spec_node, depart_spec_node),
    )

    app.add_directive("spec", SpecDirective)
    app.add_directive("assumption", AssumptionDirective)
    app.add_directive("limitation", LimitationDirective)
    app.add_directive("requirement", RequirementDirective)
    app.add_directive("speclist", SpeclistDirective)
    app.connect("doctree-resolved", process_spec_nodes)
    app.connect("env-purge-doc", purge_specs)
    app.connect("env-merge-info", merge_specs)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
