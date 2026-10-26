# Thesis — Overleaf project (Tor Vergata `tvthesis` class)

**Correction-Driven Continual Learning for Grounded Chest X-ray Reporting**
*A Gated Update Loop for a Medical Vision–Language Model Evaluated on Held-Out PadChest-GR*

Afshin Esmaeilzad Ahandani · Mechatronics Engineering · Advisor: Prof. Arianna Mencattini · A.Y. 2025/2026

| file | content |
|---|---|
| `main.tex` | **main document**: class options (English, lists of figures/tables), title page, front matter, chapter includes |
| `tvthesis.cls` | the university template class (do not edit) |
| `chapters/introduction.tex` | Chapter 1 — **first draft, for review** |
| `chapters/background.tex` … `conclusion.tex` | Chapters 2–5 — agreed outline, titles only |
| `chapters/appendix.tex` | Appendices A–B — titles only |
| `references.bib` | bibliography; every entry checked against its source |
| `figures/` | the seven thesis figures, for the later chapters |
| `example-master.tex` | the template's original Italian example — reference only, not compiled |

The source for every number and every decision is
`cxr_gui/thesis/PROJECT_DOSSIER.md` in the private `CXR-GroundAssist` repository.

## Compiling

Overleaf compiles automatically. In **Menu → Settings → Main document** choose
`main.tex` (not `example-master.tex`). Locally:
`pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Connecting Overleaf to GitHub

Overleaf's GitHub synchronisation treats **one whole repository as one
project**: the repository root must contain `main.tex`, `tvthesis.cls` and the
rest of this folder. This folder currently lives inside the `RX` repository,
so it cannot be imported as-is — give it its own repository first.

1. **Create a dedicated repository** whose root is this folder (private is
   recommended — the title page carries your name and student ID).
2. In Overleaf: **New Project → Import from GitHub**, authorise Overleaf on
   GitHub, and pick that repository. Overleaf creates the project from it.
3. From then on use **Menu → Sync → GitHub**: *Pull GitHub changes into
   Overleaf* after editing on your computer, *Push Overleaf changes to
   GitHub* after editing in Overleaf.

GitHub synchronisation is a feature of Overleaf's paid plans; some
universities provide it through an institutional licence. Without it, use
**New Project → Upload Project** with a zip of this folder, and download the
project as a zip to bring changes back.
