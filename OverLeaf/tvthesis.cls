%%
%% This is file `tvthesis.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% tvthesis.dtx  (with options: `class')
%% 
%% This is a generated file.
%% 
%% Copyright (C) 2026 by Alessandro Pellegrini
%% 
%% This file may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either
%% version 1.3c of this license or (at your option) any later
%% version. The latest version of this license is in:
%% 
%%    http://www.latex-project.org/lppl.txt
%% 
%% and version 1.3c or later is part of all distributions of
%% LaTeX version 2008-05-04 or later.
%% 
\NeedsTeXFormat{LaTeX2e}[2009/01/01]
\ProvidesClass{tvthesis}%
[2026/07/21 v1.1 University of Rome Tor Vergata thesis class]
\RequirePackage{xkeyval}
\newlength{\TV@binding}
\newif\ifTV@nodefaultfont \TV@nodefaultfontfalse
\newif\ifTV@customlanguage \TV@customlanguagefalse
\newif\ifTV@femaleauthor \TV@femaleauthorfalse
\newif\ifTV@master \TV@masterfalse
\newif\ifTV@phd \TV@phdtrue
\newif\ifTV@showcrop \TV@showcropfalse
\newif\ifTV@showmargins \TV@showmarginsfalse
\newif\ifTV@apaper \TV@apaperfalse
\newif\ifTV@letterpaper \TV@letterpaperfalse
\newif\ifTV@lof \TV@loffalse
\newif\ifTV@lot \TV@lotfalse

\newcommand{\TV@customlanguage}{}
\newcommand{\TV@CandidateText}{\ifTV@femaleauthor\TV@CandidateLabelFemale\else\TV@CandidateLabel\fi}
\define@key[tv]{}{italian}[]{\TV@customlanguagetrue
  \renewcommand{\TV@customlanguage}{\TV@italian}}
\define@key[tv]{}{english}[]{\TV@customlanguagetrue
  \renewcommand{\TV@customlanguage}{\TV@english}}
\define@key[tv]{}{fem}[]{\TV@femaleauthortrue}
\define@key[tv]{}{nodefaultfont}[]{\TV@nodefaultfonttrue}
\define@key[tv]{}{binding}[0pt]{\setlength{\TV@binding}{#1}}
\define@key[tv]{}{draft}[]{\PassOptionsToClass{draft}{book}}
\define@key[tv]{}{oneside}[]{\PassOptionsToClass{oneside}{book}}
\define@key[tv]{}{twoside}[]{}
\define@key[tv]{}{master}[]{\TV@mastertrue\TV@phdfalse}
\define@key[tv]{}{phd}[]{\TV@phdtrue\TV@masterfalse}
\define@key[tv]{}{a4paper}[]{\TV@apapertrue}
\define@key[tv]{}{letterpaper}[]{\TV@letterpapertrue}
\define@key[tv]{}{showcrop}[]{\TV@showcroptrue}
\define@key[tv]{}{showmargins}[]{\TV@showmarginstrue}
\define@key[tv]{}{lof}[]{\TV@loftrue}
\define@key[tv]{}{lot}[]{\TV@lottrue}

\DeclareOptionX*{\ClassError{tvthesis}{Option '\CurrentOption'%
  \MessageBreak is not valid}{Remove '\CurrentOption' %
  from class options}}
\ProcessOptionsX[tv]<>
\LoadClass[11pt]{book}
\RequirePackage{geometry}

\ifTV@master
  \geometry{a4paper,
    textwidth=396pt,
    textheight=641pt,
    marginparsep=12pt,
    bindingoffset=\TV@binding,
    vmarginratio=55:89,
    includehead,
    marginparwidth=72pt,
    heightrounded}
  \if@twoside\geometry{hmarginratio=55:89}\fi
\else
  \geometry{twoside,
    left=2.5cm,
    hmarginratio=2:1,
    includemp,
    marginparwidth=0pt,
    bottom=2.5cm,
    paperheight=24cm,
    paperwidth=17cm,
    layoutheight=24cm,
    layoutwidth=17cm,
    headheight=13.6pt}
  \ifTV@apaper
    \geometry{paper=a4paper,
      layouthoffset=\dimexpr0.5\paperwidth-0.5\Gm@layoutwidth,
      layoutvoffset=\dimexpr0.5\paperheight-0.5\Gm@layoutheight}
  \fi
  \ifTV@letterpaper
    \geometry{paper=letterpaper,
      layouthoffset=\dimexpr0.5\paperwidth-0.5\Gm@layoutwidth,
      layoutvoffset=\dimexpr0.5\paperheight-0.5\Gm@layoutheight}
  \fi
\fi

\ifTV@showcrop\geometry{showcrop}\fi
\ifTV@showmargins\geometry{showframe}\fi
\RequirePackage{ifxetex}
\RequirePackage{amsmath}

\ifxetex
  \RequirePackage{fontspec}
\else
  \RequirePackage[utf8]{inputenc}
  \ifTV@nodefaultfont\relax\else
    \RequirePackage[T1]{fontenc}
    \RequirePackage{textcomp}
    \RequirePackage{libertine}
    \RequirePackage[defaultsans]{cantarell}
    \RequirePackage[scaled=0.96]{inconsolata}
    \RequirePackage{libertinust1math}
  \fi
\fi

\RequirePackage{graphicx}
\RequirePackage{color}
\RequirePackage{booktabs}
\RequirePackage{etoolbox}
\RequirePackage{tikz}
\RequirePackage{setspace}
\RequirePackage{microtype}
\RequirePackage{titlesec}
\RequirePackage{enumitem}
\setlist{noitemsep}
\RequirePackage{caption}
\captionsetup{indention=\parindent,labelfont={small,bf},labelsep=period,%
  textfont={small}}
\RequirePackage{floatrow}
\floatsetup[figure]{capposition=bottom}
\floatsetup[table]{capposition=top}
\renewcommand{\baselinestretch}{1.5}
\definecolor{verdetv}{RGB}{0,125,52}
\definecolor{nerotv}{RGB}{29,29,27}
\newcommand{\tower}{%
\begin{tikzpicture}
    \draw[fill=verdetv,draw=white] (-1.85,-3) rectangle (2.4,2.8);
    \filldraw[fill=white,draw=white] (-1.85,2.8) -- (1,2.8) -- (1,0) -- (-1.85,0);
    \filldraw[fill=white,draw=white] (1,0) arc (360:240:1.9) -- (-1.85,0);
    \def\TV@towerScaleA{0.45}%
    \foreach \i in {0,...,6} {
        \draw[fill=verdetv,draw=verdetv] (-1.85,-0.55+\TV@towerScaleA*\i)
          rectangle (-0.05,-0.3+\TV@towerScaleA*\i);
    }
    \def\TV@towerScaleB{0.4}%
    \foreach \i in {0,...,4}
        \draw[fill=verdetv,draw=verdetv] (-1.85+\TV@towerScaleB*\i,2.65)
          rectangle (-1.65+\TV@towerScaleB*\i,2.8);
\end{tikzpicture}
}
\newsavebox\TV@towerbox
\sbox\TV@towerbox{%
    \begin{minipage}[b]{1.8cm}%
        \resizebox{\linewidth}{!}{\tower}%
    \end{minipage}%
}
\newcommand{\TV@rendertitlelogo}{%
  \sbox0{\fontsize{30}{36}\selectfont\textcolor{verdetv}{TOR VERGATA}}%
  \sbox2{%
    \begin{tabular}[b]{@{}l@{}}
      \usebox0 \\[1pt]
      \resizebox{\wd0}{0.5\ht0}{%
        \textcolor{verdetv}{UNIVERSIT\`A DEGLI STUDI DI ROMA}}
    \end{tabular}%
  }%
  \dimen0=1.3\dimexpr\ht2+\dp2\relax%
  \sbox4{\resizebox{!}{\dimen0}{\usebox\TV@towerbox}}%
  \dimen2=\dimexpr(\ht4+\dp4-\ht2-\dp2)/2+4pt\relax%
  \usebox4%
  \hspace{1.5em}%
  \raisebox{\dimen2}{\usebox2}%
}
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
\thispagestyle{empty}\hbox{}\newpage\fi\fi}

\RequirePackage{fancyhdr}
\fancypagestyle{TV@mainstyle}{%
\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\markboth{%
  \ifnum \c@chapter>0 \thechapter.\ ##1 \else ##1 \fi}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection\ ##1}}
\fancyhf{}
\fancyhead[OR]{\small\bfseries\thepage}
\fancyhead[OL]{\small\bfseries\nouppercase{\rightmark}}
\if@twoside
\fancyhead[ER]{\small\bfseries\nouppercase{\leftmark}}
\fancyhead[EL]{\small\bfseries\thepage}
\fi
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}}
\fancypagestyle{plain}{%
\renewcommand{\headrulewidth}{0pt}
\fancyhf{}}
\numberwithin{equation}{chapter}
\numberwithin{figure}{chapter}
\numberwithin{table}{chapter}

\titleformat{\chapter}[display]
{\bfseries\Large\filleft}
{\filleft\scshape{\chaptertitlename} \thechapter}
{6pt}
{\Huge\bfseries}
[\vspace{25pt}\titlerule]
\def\TV@subtitle{}
\def\TV@alttitle{}
\newcommand{\subtitle}[1]{\def\TV@subtitle{#1}}
\newcommand{\alttitle}[1]{\def\TV@alttitle{#1}}

\def\TV@courseorganizerLabel{}
\def\TV@courseLabel{}
\def\TV@thesissubject{}
\newcommand{\courseorganizer}[1]{\def\TV@courseorganizerLabel{#1}}
\newcommand{\course}[1]{\def\TV@courseLabel{#1}}
\newcommand{\thesissubject}[1]{\def\TV@thesissubject{#1}}

\def\TV@doctorate{}
\newcommand{\doctorate}[1]{\def\TV@doctorate{#1}}

\def\TV@cycle{}
\newcommand{\cycle}[1]{\def\TV@cycle{#1}}

\def\TV@AcademicYear{}
\newcommand{\AcademicYear}[1]{\def\TV@AcademicYear{#1}}
\let\submitdate\AcademicYear

\def\TV@IDnumber{}
\newcommand{\IDnumber}[1]{\def\TV@IDnumber{#1}}
\newcount\TV@advisorcount
\newtoks\TV@advisortoks
\newcommand{\advisor}[1]{%
  \ifnum\TV@advisorcount=\z@
    \TV@advisortoks={#1}%
  \else
    \TV@advisortoks=\expandafter{\the\TV@advisortoks\\[1mm]#1}%
  \fi
  \advance\TV@advisorcount\@ne}

\newcommand{\customadvisorlabel}[1]{%
    \newcommand{\TV@ThesisAdvisorLabel}{#1}%
    \newcommand{\TV@ThesisAdvisorsLabel}{#1}}

\newcount\TV@coadvisorcount
\newtoks\TV@coadvisortoks
\newcommand{\coadvisor}[1]{%
  \ifnum\TV@coadvisorcount=\z@
    \TV@coadvisortoks={#1}%
  \else
    \TV@coadvisortoks=\expandafter{\the\TV@coadvisortoks\\[1mm]#1}%
  \fi
  \advance\TV@coadvisorcount\@ne}

\newcommand{\customcoadvisorlabel}[1]{%
    \newcommand{\TV@ThesisCoAdvisorLabel}{#1}%
    \newcommand{\TV@ThesisCoAdvisorsLabel}{#1}}

\newcount\TV@reviewercount
\newtoks\TV@reviewertoks
\newcommand{\reviewer}[1]{%
  \ifnum\TV@reviewercount=\z@
    \TV@reviewertoks={#1}%
  \else
    \TV@reviewertoks=\expandafter{\the\TV@reviewertoks\\[1mm]#1}%
  \fi
  \advance\TV@reviewercount\@ne}

\newcount\TV@committeecount
\newtoks\TV@committeetoks
\newcommand{\committeemember}[1]{%
  \ifnum\TV@committeecount=\z@
    \TV@committeetoks={#1}%
  \else
    \TV@committeetoks=\expandafter{\the\TV@committeetoks\\[1mm]#1}%
  \fi
  \advance\TV@committeecount\@ne}

\newcount\TV@directorcount
\newtoks\TV@directortoks
\newcommand{\director}[1]{%
  \ifnum\TV@directorcount=\z@
    \TV@directortoks={#1}%
  \else
    \TV@directortoks=\expandafter{\the\TV@directortoks\\[0.75mm]#1}%
  \fi
  \advance\TV@directorcount\@ne}

\newcommand{\customdirectorlabel}[1]{%
    \newcommand{\TV@ThesisDirectorLabel}{#1}%
    \newcommand{\TV@ThesisDirectorsLabel}{#1}}

\def\TV@authoremail{}
\newcommand{\authoremail}[1]{\def\TV@authoremail{#1}}

\def\TV@authoraddress{}
\newcommand{\authoraddress}[1]{\def\TV@authoraddress{#1}}

\def\TV@website{}
\newcommand{\website}[1]{\def\TV@website{#1}}
\newcommand{\remove}[1]{}
\providecommand*{\dedication}[1]{%
\cleardoublepage
\thispagestyle{empty}
\null\vspace{\stretch{0.26}}
\begin{flushright}
\textit{#1}
\end{flushright}
\vspace{\stretch{0.74}}\null}

\newenvironment{abstract}[1][\TV@AbstractLabel]
{\cleardoublepage\thispagestyle{plain}
\noindent{\Large\bfseries #1}\par\vspace{0.5cm}\noindent\ignorespaces}
{\null\ignorespacesafterend}

\newenvironment{acknowledgments}[1][\TV@AcknowledgmentsLabel]
{\cleardoublepage\thispagestyle{plain}
\noindent{\Large\bfseries #1}\par\vspace{0.5cm}\noindent\ignorespaces}
{\null\ignorespacesafterend}
\g@addto@macro\frontmatter{\pagestyle{plain}}
\g@addto@macro\mainmatter{\pagestyle{TV@mainstyle}}

\let\TV@oldtableofcontents\tableofcontents
\renewcommand{\tableofcontents}{%
  \TV@oldtableofcontents
  \ifTV@lof\listoffigures\fi
  \ifTV@lot\listoftables\fi
}
\let\TV@oldbibliography\bibliography
\renewcommand{\bibliography}[1]{%
  \markboth{\textsc{\bibname}}{\textsc{\bibname}}
  \begin{small}
  \begin{spacing}{1}
  \phantomsection
  \addcontentsline{toc}{chapter}{\bibname}
  \TV@oldbibliography{#1}
  \end{spacing}
  \end{small}
}
\RequirePackage[%
    pagebackref,
    plainpages=false,
    hypertexnames=true,
    pdfpagelabels,
    bookmarks,
    bookmarksopen=false,
    bookmarksnumbered=true,
    breaklinks=true,
    linktocpage,
    colorlinks=false,
    hidelinks,
    hyperindex=true]{hyperref}

\renewcommand*{\backreflastsep}{, }
\renewcommand*{\backreftwosep}{, }
\renewcommand*{\backref}[1]{}
\renewcommand*{\backrefalt}[4]{%
  \ifcase #1 %
    \relax
  \or
    (cited on page~#2.)%
  \else
    (cited on pages #2.)%
  \fi%
}
\newcommand{\TV@maketitlePhD}{%
\pagestyle{empty}%
\newgeometry{hmarginratio=1:1}%
\def\TV@fs@normal{\@setfontsize\TV@fs@normal\@xipt{10}}%
\def\TV@fs@large{\@setfontsize\TV@fs@large\@xiipt{14}}%
\def\TV@fs@Large{\@setfontsize\TV@fs@Large\@xivpt{18}}%
\def\TV@fs@LARGE{\@setfontsize\TV@fs@LARGE\@xviipt{20}}%
%%% Cover Page
\vspace*{5mm}
\begin{center}
  \TV@rendertitlelogo
  \vspace{10mm}

  \textsc{{\TV@fs@large \TV@DottoratoHeader}\\[2mm]%
  \TV@cycle\ \TV@cycleLabel}\\[5mm]

  \vfill
  {\TV@fs@Large \textbf{\@title}\par}
  \if|\TV@subtitle|\relax\else
    \vspace{0.4cm}
    {\TV@fs@large\TV@subtitle\par}
  \fi
  \vfill

  {\TV@fs@Large \@author}

  \vfill
  \textsc{{\TV@fs@large \TV@AcademicYear}}
\end{center}
\cleardoublepage
%%% Internal Page
\vspace*{5mm}
\begin{center}
  \TV@rendertitlelogo
  \vspace{10mm}

  \textsc{{\TV@fs@large \TV@DottoratoHeader\\[2mm]%
  \TV@cycle\ \TV@cycleLabel}}\\[5mm]

  \vfill
  {\TV@fs@Large \@author \par}
  \vfill

  {\TV@fs@Large \textbf{\@title} \par}
  \if|\TV@subtitle|\relax\else
    \vspace{0.3cm}
    {\TV@fs@large\TV@subtitle\par}
  \fi
  \vspace*{0.5cm}

  \begin{tabular}[t]{c@{\hspace{10mm}}c}
    \begin{tabular}[t]{l}
      \multicolumn{1}{c}{\textbf{\TV@CommitteeHeader}}\\[5mm]
      \the\TV@advisortoks\ (\TV@AdvisorHeaderLabel) \\
      \ifnum\TV@committeecount>\z@
        \the\TV@committeetoks \\
      \fi
    \end{tabular}%
    &%
    \ifnum\TV@reviewercount>\z@
      \begin{tabular}[t]{l}
        \multicolumn{1}{c}{\textbf{\TV@ReviewerHeaderLabel}}\\[5mm]
        \the\TV@reviewertoks \\
      \end{tabular}
    \fi
  \end{tabular}

  \vfill
  \textsc{{\TV@fs@large \TV@AcademicYear}}
\end{center}
\clearpage
\restoregeometry
%%% Author Address Page
\begin{raggedright}
\vspace*{\fill}
{\TV@fs@normal
  \textsc{\TV@AddressHeader} \\
  \@author \\
  \TV@authoraddress \\
  \textsc{e-mail:} \texttt{\TV@authoremail}\\
  \if|\TV@website|\relax\else\textsc{www:} \texttt{\TV@website}\fi}\par
\vspace*{2cm}
\end{raggedright}
\clearpage
}
\newcommand{\TV@maketitleMaster}{%
\clearpage
\thispagestyle{empty}%
\newgeometry{hmarginratio=1:1}

\linespread{1}\selectfont
\parindent=0pt

\begin{center}
{\bfseries\large UNIVERSIT\`A DEGLI STUDI DI ROMA TOR VERGATA}\\[2mm]
{\large\scshape \TV@courseorganizerLabel}
\end{center}

\vspace{4mm}

\begin{center}
\TV@rendertitlelogo
\end{center}

\vspace{4mm}

\begin{center}
{\scshape \TV@CorsoDiStudioHeader}\\[1mm]
{\large\MakeUppercase{\TV@courseLabel}}%
\if|\TV@thesissubject|\relax\else
  \\[3mm]
  {\scshape \TV@TesiDiLaureaInHeader}\\[1mm]
  {\large \TV@thesissubject}%
\fi
\end{center}

\vspace{\stretch{1}}

\begin{center}
{\LARGE\bfseries\linespread{1.1}\selectfont\@title\par}
\if|\TV@subtitle|\relax\else
  \vspace{0.4cm}
  {\large\linespread{1.1}\selectfont\TV@subtitle\par}
\fi
\if|\TV@alttitle|\relax\else
  \vspace{0.8cm}
  {\LARGE\linespread{1.1}\selectfont\TV@alttitle\par}
\fi
\end{center}

\vspace{\stretch{1}}

\begin{center}
\makebox[\textwidth][c]{%
  \begin{tabular}{@{}l@{\hspace{4.0cm}}l@{}}
    \begin{tabular}[t]{@{}l@{}}
      \ifcase\TV@advisorcount
        \ClassError{tvthesis}{You must specify at least one advisor in the preamble}%
        {E.g. \protect\advisor{Prof. Name Surname}}
      \or
        \textbf{\TV@ThesisAdvisorLabel:}\\[1mm]
      \else
        \textbf{\TV@ThesisAdvisorsLabel:}\\[1mm]
      \fi
      \the\TV@advisortoks
      \ifnum\TV@coadvisorcount>\z@
        \\[4mm]
        \ifnum\TV@coadvisorcount=1
          \textbf{\TV@ThesisCoAdvisorLabel:}\\[1mm]
        \else
          \textbf{\TV@ThesisCoAdvisorsLabel:}\\[1mm]
        \fi
        \the\TV@coadvisortoks
      \fi
    \end{tabular}%
    &%
    \begin{tabular}[t]{@{}l@{}}
      \ifTV@femaleauthor
        \textbf{\TV@CandidateLabelFemale:}\\[1mm]
      \else
        \textbf{\TV@CandidateLabel:}\\[1mm]
      \fi
      {\large\@author}\\[1mm]
      \TV@IDnumberLabel\ \TV@IDnumber
    \end{tabular}%
  \end{tabular}%
}
\end{center}

\vspace{\stretch{2}}

\begin{center}
\TV@AcademicYearLabel\ \TV@AcademicYear
\end{center}

\restoregeometry
\clearpage
}
\renewcommand{\maketitle}{%
\pagenumbering{Alph}%
\if|\TV@AcademicYear|\ClassError{tvthesis}{You must specify the %
  academic year}{Example: \protect\AcademicYear{2024/2025}}\fi
\ifTV@master
  \if|\TV@courseorganizerLabel|\ClassError{tvthesis}{You must specify %
    the course organizer}{Example: %
    \protect\courseorganizer{Facolt\`a di Ingegneria}}\fi
  \if|\TV@courseLabel|\ClassError{tvthesis}{You must specify the %
    course}{Example: \protect\course{Corso di Laurea Magistrale %
    in Informatica}}\fi
  \if|\TV@IDnumber|\ClassError{tvthesis}{You must specify your %
    ID number (matricola)}{Example: \protect\IDnumber{0123456}}\fi
  \TV@maketitleMaster
\else
  \if|\TV@doctorate|\ClassError{tvthesis}{You must specify the %
    doctorate program}{Example: \protect\doctorate{Computer Science}}\fi
  \if|\TV@cycle|\ClassError{tvthesis}{You must specify the %
    cycle}{Example: \protect\cycle{XXXVIII}}\fi
  \if|\TV@authoremail|\ClassError{tvthesis}{You must specify the %
    author's email}{Example: \protect\authoremail{author@email.com}}\fi
  \TV@maketitlePhD
\fi
}
\newcommand{\TV@english}{%
  \def\TV@CandidateLabel{Candidate}%
  \def\TV@CandidateLabelFemale{Candidate}%
  \def\TV@IDnumberLabel{ID number:}%
  \def\TV@CorsoDiStudioHeader{Degree Programme in}%
  \def\TV@TesiDiLaureaInHeader{Thesis in}%
  \providecommand{\TV@ThesisAdvisorLabel}{Advisor}%
  \providecommand{\TV@ThesisAdvisorsLabel}{Advisors}%
  \providecommand{\TV@ThesisCoAdvisorLabel}{Co-Advisor}%
  \providecommand{\TV@ThesisCoAdvisorsLabel}{Co-Advisors}%
  \providecommand{\TV@ThesisDirectorLabel}{Director}%
  \providecommand{\TV@ThesisDirectorsLabel}{Directors}%
  \def\TV@cycleLabel{Cycle}%
  \def\TV@AcademicYearLabel{Academic Year}%
  \def\TV@universityname{University of Rome Tor Vergata}%
  \def\TV@AcknowledgmentsLabel{Acknowledgments}%
  \def\TV@AbstractLabel{Abstract}%
  \def\TV@DottoratoHeader{Ph.D.\ program in \TV@doctorate}%
  \def\TV@CommitteeHeader{Thesis Committee}%
  \def\TV@AdvisorHeaderLabel{Advisor}%
  \def\TV@ReviewerHeaderLabel{Reviewers}%
  \def\TV@AddressHeader{Author's address:}%
  \def\contentsname{Contents}%
  \def\bibname{Bibliography}%
  \def\listfigurename{List of Figures}%
  \def\listtablename{List of Tables}%
  \def\chaptername{Chapter}%
  \def\appendixname{Appendix}%
  \def\figurename{Figure}%
  \def\tablename{Table}%
}

\newcommand{\TV@italian}{%
  \def\TV@CandidateLabel{Laureando}%
  \def\TV@CandidateLabelFemale{Laureanda}%
  \def\TV@IDnumberLabel{matricola:}%
  \def\TV@CorsoDiStudioHeader{Corso di Studio in}%
  \def\TV@TesiDiLaureaInHeader{Tesi di Laurea in}%
  \providecommand{\TV@ThesisAdvisorLabel}{Relatore}%
  \providecommand{\TV@ThesisAdvisorsLabel}{Relatori}%
  \providecommand{\TV@ThesisCoAdvisorLabel}{Correlatore}%
  \providecommand{\TV@ThesisCoAdvisorsLabel}{Correlatori}%
  \providecommand{\TV@ThesisDirectorLabel}{Direttore}%
  \providecommand{\TV@ThesisDirectorsLabel}{Direttori}%
  \def\TV@cycleLabel{Ciclo}%
  \def\TV@AcademicYearLabel{Anno Accademico}%
  \def\TV@universityname{Universit\`a degli Studi di Roma Tor Vergata}%
  \def\TV@AcknowledgmentsLabel{Ringraziamenti}%
  \def\TV@AbstractLabel{Sommario}%
  \def\TV@DottoratoHeader{Dottorato di Ricerca in \TV@doctorate}%
  \def\TV@CommitteeHeader{Commissione di Tesi}%
  \def\TV@AdvisorHeaderLabel{Relatore}%
  \def\TV@ReviewerHeaderLabel{Revisori}%
  \def\TV@AddressHeader{Indirizzo dell'autore:}%
  \def\contentsname{Indice}%
  \def\bibname{Bibliografia}%
  \def\listfigurename{Elenco delle figure}%
  \def\listtablename{Elenco delle tabelle}%
  \def\chaptername{Capitolo}%
  \def\appendixname{Appendice}%
  \def\figurename{Figura}%
  \def\tablename{Tabella}%
}
\AtBeginDocument{%
  \ifTV@customlanguage
    \TV@customlanguage
  \else
    \@ifpackageloaded{babel}{\csname TV@\languagename\endcsname}{%
      \@ifpackageloaded{polyglossia}{\csname TV@\languagename\endcsname}{%
        \TV@english}}
  \fi
}

\endinput
%%
%% End of file `tvthesis.cls'.
