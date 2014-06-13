\name{DimSqueezer}
\Rdversion{1.1}
\docType{class}
\alias{Rcpp_DimSqueezer-class}
\alias{DimSqueezer}

\title{Class \code{"Rcpp_DimSqueezer"}}
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
  \url{http://martin.jakt.org.uk/}
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
\code{\linkS4class{Rcpp_DimSqueezer_CL}}\cr
\code{\link{serial.dim.factors}}\cr
\code{\link{parallel.dim.factors}}\cr
\code{\link{parallel.exp.dim.factors}}
}

%% we don't run the examples as the same example is given in the SOD
%% package
\examples{
\dontrun{
## get the attached data f186
data(f186)
## columns 7:12 contain data of the same type
## see the documentation for f186

## reduce the number of objects for automated checks
r <- 1:200
ds <- new(DimSqueezer, as.matrix(f186[r,7:12]))

sq <- ds$squeeze(2, 200)

plt.points(sq)

## or for a fancy plot
plt.concentric(sq, cex.data=f186[r,7:12], leg.pos='topleft')

## to see the stress evolution
plt.stress(sq)

}
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
