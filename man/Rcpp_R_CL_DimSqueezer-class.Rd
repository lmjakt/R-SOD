\name{R_CL_DimSqueezer}
\Rdversion{1.1}
\docType{class}
\alias{Rcpp_R_CL_DimSqueezer-class}
\alias{R_CL_DimSqueezer}

\title{Class \code{"Rcpp_R_CL_DimSqueezer"}}
\description{
  %%  ~~ A concise (1-5 lines) description of what the class is. ~~
  Provides an S4 class that implements the self-organising-deltoids
  (SOD) multi-dimensional-scaling algorithm (MDS) using openCL to
  run the algorithm on a graphics processing unit. This is only available on computers having the appropriate hardware
  and will fail reporting errors if a suitable GPU and libraries are not present.
}
\section{Extends}{
Class \code{"\linkS4class{C++Object}"}, directly.

All reference classes extend and inherit methods from \code{"\linkS4class{envRefClass}"}.

}
\references{
  %%  ~~put references to the literature/web site here~~
  \url{http://martin.jakt.org.uk/sod/}
}
\author{
  %%  ~~who you are~~
  Martin Jakt
}
%%\note{
%%  ~~further notes~~
%%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%%  ~~objects to See Also as \code{\link{~~fun~~}}, ~~~
%%  ~~or \code{\linkS4class{CLASSNAME}} for links to other classes ~~~
%% the following give warnings, but never mind.
\code{\linkS4class{Rcpp_R_DimSqueezer}}\cr
\code{\link{serial.dim.factors}}\cr
\code{\link{parallel.dim.factors}}\cr
\code{\link{parallel.exp.dim.factors}}
}

\examples{
require(SOD)
showClass("Rcpp_R_CL_DimSqueezer")

## read in the f186 data set
data("f186", package="SOD")

## f186 contains information about a set of cells. We
## want to use columns 7:12, which contain transcript
## densities to visualise the relationships between the
## cells.

ds2 <- new(R_CL_DimSqueezer, as.matrix(f186[,7:12]) )

## squeeze, 1000 iterations using a work group size of 64
## giving a 2 dimensional representation.
sq2 <- ds2$squeeze(2, 1000, 64)

}

\keyword{classes}
\section{Methods}{
  \describe{
    \item{\code{initialize(...)}:}{}
    \item{\code{finalize()}:}{ ~~ }
    \item{\code{squeeze(target_dim, iter_no, local_work_size)}:}{
      target_dim:      the number of dimensions in the final
      representation\cr\cr
      iter_no:         the number iterations within which the dimensions are
      reduced to the target dimensionality\cr\cr
      local_work_size: a measure of the parallelism of the GPU
      processing. Usually set to 32.
    }
  }
}

