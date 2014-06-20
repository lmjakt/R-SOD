\name{SOD example data: f186}
\docType{data}
\alias{f186}
\title{Descriptions of cells}

\description{
  This data set provides descriptions of 874 cells obtained by
  microscopic analysis of cells carrying a dox inducible
  EGFP-Etv2 fusion transcription factor.\cr
  There are 21 columns indicating:
  \itemize{
    \item dox: whether EGFP-Etv2 was induced by dox (1) or not (0).
    \item time: time of induction.
    \item image: an identifier of the image set from which the data row
    was obtained.
    \item cell: an identifier for the cell.
    \item c_area: areas of cells.
    \item n_area: area of cell nuclei.
    \item Fli1, Cdh5, Flk1, Etv2, Pdgfra, Snail1, FP (false
    positives). Densities of the indicated transcripts (i.e. transcript
    no / cell area).
    \item n_Fli1, n_Cdh5, n_Flk1, n_Etv2, n_Pdgfra, n_Snail1, n_FP. The number
    of sites of active transcription for the indicated transcripts.
    \item p_Etv2. The amount of nuclear EGFP-Etv2 protein estimated from
    direct EGFP fluorescence.
  }
  The f186 name is the experimental identifier and is used here to make
  the data source easy to trace.
}

\usage{data(f186)}
\format{A dataframe containing 876 rows and 21 columns.}
\source{ Unpublished data obtained from Martin Jakt }

\keyword{dataset}
