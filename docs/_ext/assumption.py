from docutils import nodes  # type: ignore
from docutils.parsers.rst import Directive  # type: ignore
from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective


class Assumption(nodes.Admonition, nodes.Element):
    pass


class AssumptionList(nodes.General, nodes.Element):
    pass


def visit_assumption_node(self, node):
    self.visit_admonition(node)


def depart_assumption_node(self, node):
    self.depart_admonition(node)


class AssumptionlistDirective(Directive):
    def run(self):
        return [AssumptionList("")]


class AssumptionDirective(SphinxDirective):
    # this enables content in the directive
    has_content = True

    def run(self):
        targetid = "assumption-%d" % self.env.new_serialno("assumption")
        targetnode = nodes.target("", "", ids=[targetid])

        assumption_node = Assumption("\n".join(self.content))
        assumption_node += nodes.title(_("Assumption"), _("Assumption"))
        self.state.nested_parse(self.content, self.content_offset, assumption_node)

        if not hasattr(self.env, "assumption_all_assumptions"):
            setattr(self, "env.assumptions_all_assumptions", [])
        assert isinstance(self.env.assumptions_all_assumptions, list)  # type: ignore[attr-defined]

        self.env.assumptions_all_assumptions.append(  # type: ignore[attr-defined]
            {
                "docname": self.env.docname,
                "lineno": self.lineno,
                "assumption": assumption_node.deepcopy(),
                "target": targetnode,
            }
        )

        return [targetnode, assumption_node]


def purge_assumptions(app, env, docname):
    if not hasattr(env, "assumption_all_assumptions"):
        return

    env.assumption_all_assumptions = [
        assumption for assumption in env.assumption_all_assumptions if assumption["docname"] != docname
    ]


def merge_assumptions(app, env, docnames, other):
    if not hasattr(env, "assumption_all_assumptions"):
        env.assumption_all_assumptions = []
    if hasattr(other, "assumption_all_assumptions"):
        env.assumption_all_assumptions.extend(other.assumption_all_assumptions)


def process_assumption_nodes(app, doctree, fromdocname):
    if not app.config.assumption_include_assumptions:
        for node in doctree.findall(Assumption):
            node.parent.remove(node)

    # Replace all assumptionlist nodes with a list of the collected assumptions.
    # Augment each assumption with a backlink to the original location.
    env = app.builder.env

    if not hasattr(env, "assumption_all_assumptions"):
        env.assumption_all_assumptions = []

    for node in doctree.findall(AssumptionList):
        if not app.config.assumption_include_assumptions:
            node.replace_self([])
            continue

        content = []

        for assumption_info in env.assumption_all_assumptions:
            para = nodes.paragraph()
            filename = env.doc2path(assumption_info["docname"], base=None)
            description = _("(The original entry is located in %s, line %d and can be found ") % (
                filename,
                assumption_info["lineno"],
            )
            para += nodes.Text(description)

            # Create a reference
            newnode = nodes.reference("", "")
            innernode = nodes.emphasis(_("here"), _("here"))
            newnode["refdocname"] = assumption_info["docname"]
            newnode["refuri"] = app.builder.get_relative_uri(fromdocname, assumption_info["docname"])
            newnode["refuri"] += "#" + assumption_info["target"]["refid"]
            newnode.append(innernode)
            para += newnode
            para += nodes.Text(".)")

            # Insert into the assumptionlist
            content.append(assumption_info["assumption"])
            content.append(para)

        node.replace_self(content)


def setup(app):
    app.add_config_value("assumption_include_assumptions", True, "html")

    app.add_node(AssumptionList)
    app.add_node(
        Assumption,
        html=(visit_assumption_node, depart_assumption_node),
        latex=(visit_assumption_node, depart_assumption_node),
        text=(visit_assumption_node, depart_assumption_node),
    )

    app.add_directive("assumption", AssumptionDirective)
    app.add_directive("assumptionlist", AssumptionlistDirective)
    app.connect("doctree-resolved", process_assumption_nodes)
    app.connect("env-purge-doc", purge_assumptions)
    app.connect("env-merge-info", merge_assumptions)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
