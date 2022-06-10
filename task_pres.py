from pathlib import Path

import pytask


from config import revealjs_pandoc


name = "software-pres"


@pytask.mark.parametrize(
    "depends_on, produces",
    [
        (
            [f"{name}.md", "revealjs/template-revealjs.html", "config.py"]
            + list((Path(__file__).parent / "figures").glob("*.*")),
            f"{name}.html",
        )
    ],
)
def task_convert_revealjs(depends_on, produces):
    return revealjs_pandoc(depends_on, produces)
