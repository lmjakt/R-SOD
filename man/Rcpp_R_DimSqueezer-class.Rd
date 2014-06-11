\name{R_DimSqueezer}
\Rdversion{1.1}
\docType{class}
\alias{Rcpp_R_DimSqueezer-class}
\alias{R_DimSqueezer}

\title{Class \code{"Rcpp_R_DimSqueezer"}}
\description{
  %%  ~~ A concise (1-5 lines) description of what the class is. ~~
  Provides an S4 class that implements the self-organising-deltoids
  (SOD) multi-dimensional-scaling algorithm (MDS) that uses openMP to
  provide parallel execution on multiple CPU cores.    
}
\section{Extends}{
Class \code{"\linkS4class{C++Object}"}, directly.

All reference classes extend and inherit methods from \code{"\linkS4class{envRefClass}"}.

}
\references{
  %%  ~~put references to the literature/web site here~~
  \url{http://martin.jakt.org.uk}
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
\code{\linkS4class{Rcpp_R_CL_DimSqueezer}}\cr
\code{\link{serial.dim.factors}}\cr
\code{\link{parallel.dim.factors}}\cr
\code{\link{parallel.exp.dim.factors}}
}
\examples{
showClass("Rcpp_R_DimSqueezer")
}
\keyword{classes}
\section{Methods}{
  \describe{
    \item{\code{initialize(...)}:}{ ~~ }
    \item{\code{finalize()}:}{ ~~ }
    \item{\code{useOpenMP(...)}:}{ ~~ }
    \item{\code{squeezeDF(...)}:}{ ~~ }
    \item{\code{squeeze(...)}:}{ ~~ }
  }
}
