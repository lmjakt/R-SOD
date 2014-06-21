\name{SOD.dim.functions}

\alias{parallel.dim.factors}
\alias{parallel.exp.dim.factors}
\alias{serial.dim.factors}

\title{Specifying SOD dimensionality}

\description{
  Functions that provide a matrix of dimensionality factors that can be
  used by the \code{R_DimSqueezer} \code{squeezeDF()} function.\cr
  During the mapping process the dimensionality is gradually
  reduced. The dimensionality at a given time is represented by a vector
  of values (one for each dimension) lying within the range 0-1. 0
  indicates a null dimension; i.e. coordinates within that dimension
  have no impact upon distances between objects. A value of 1 indicates
  a full dimension.\cr
  These functions return matrices that define the dimensionality with
  one row for each iteration of the mapping process.
  \itemize{
    \item
    \code{parallel.dim.factors},\cr \code{parallel.exp.dim.factors}\cr 
    Reduce the superfluous dimensions in parallel, using a linear or
    exponential decrease respectively.
    \item
    \code{serial.dim.factors}\cr Reduces superfluous dimensions serially one
    after the other.
  }
  \if{latex}{
    \figure{parSerialStress.pdf}\cr
    Reduction of dimensions using parallel (left)
    or serial (right) strategies. The filled polygons represent the
    dimensionality, starting (left) with a fully 6-dimensional space
    that is gradually reduced to a two dimensions during the mapping
    process.\cr
    Both \code{squeeze} and \code{squeezeDF} will continue iterating
    after the specified iteration number until a minimum is reached.
  }
}

\usage{
parallel.dim.factors(dim, iteration.no,
                     red.end=iteration.no*0.75, target.dim=2)

parallel.exp.dim.factors(dim, iteration.no,
                         target.dim=2, red.end=iteration.no * 0.9)

serial.dim.factors(dim, iteration.no,
                   red.end=iteration.no*0.75, target.dim=2)

}

\arguments{
  \item{dim}{The starting number of dimensions}
  \item{target.dim}{The target number of dimensions}
  \item{iteration.no}{The total number of iterations for the mapping
    procedure}
  \item{red.end}{The number of the iterations which should be
    used to remove superflous dimensions. Iterations after this are used
    to adjust the final positions.}
}

\value{
  Returns a numeric matrix with \code{iteration.no} rows and \code{dim}
  columns.
}



\examples{
## reduce to two dimensions using (0.75 * 200) dimensions
## 
df.p <- parallel.dim.factors(6, 200, target.dim=2)

## start with 6 dimensions, and reduce over 175 iterations
## to 2 dimensions.
df.pe <- parallel.exp.dim.factors(6, 200, 175, target.dim=2)

}
