\name{DimSqueezer-class}
\Rdversion{1.1}
\docType{class}
\alias{DimSqueezer-class}
\alias{DimSqueezer}

\title{Class \code{"DimSqueezer"}}
\description{
  %%  ~~ A concise (1-5 lines) description of what the class is. ~~
  Provides a reference class that implements the self-organising-deltoids
  (SOD) multi-dimensional-scaling algorithm (MDS) using openMP to
  provide parallel execution on multiple CPU cores.\cr
  New objects are made by calling the initializer using either
  \code{DimSqueezer$new(m)} or \code{new("DimSqueezer", m)}.\cr
  The initializer returns an object from which the methods are called
  using the \code{$} operator. See the examples section for usage.
}
\section{Extends}{

All reference classes extend and inherit methods from \code{"\linkS4class{envRefClass}"}.

}
\references{
  %%  ~~put references to the literature/web site here~~
  \url{http://martin.jakt.org.uk/r-sod/}
}
\author{
  %%  ~~who you are~~
  Martin Jakt
}
%%\note{
%%  ~~further notes~~
%%}

\section{Warning}{
  This is a reference class that does not follow normal R semantic
  rules. Copying objects does not make new copies, but additional
  references. See \code{?setRefClass} for more details.
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%%  ~~objects to See Also as \code{\link{~~fun~~}}, ~~~
%%  ~~or \code{\linkS4class{CLASSNAME}} for links to other classes ~~~
\code{\linkS4class{DimSqueezer_CL}}\cr
\code{\link{serial.dim.factors}}\cr
\code{\link{parallel.dim.factors}}\cr
\code{\link{parallel.exp.dim.factors}}
}

\examples{
showClass("DimSqueezer")
## get the attached data f186
data(f186)
## columns 7:12 contain data of the same type
## see the documentation for f186
m <- as.matrix(f186[,7:12])

## reduce the number of objects to speed up automated
## checks.
r <- 1:200
ds <- new("DimSqueezer", m[r,])
## equivalent to
## ds <- DimSqueezer$new( m[r,] )

sq <- ds$squeeze(2, 200)

plt.points(sq)
## with more informative colours
plt.points(sq, col=hsv.scale(f186[,"Etv2"]), pch=19, cex=1.5)

## or for a fancy plot
plt.concentric(sq, cex.data=f186[r,7:12], leg.pos="topleft")

## to see the stress evolution
plt.stress(sq)
}

\keyword{classes}
\section{Fields}{
  \describe{
    \item{\code{pointer}:}{Object of class \code{externalptr}
      that refers to the C++ DimSqueezer object (read only).}
    \item{\code{data.matrix}:}{Object of class \code{matrix}
      a numeric matrix used to create the DimSqueezer.}
  }
}

\section{Methods}{
  \describe{
    \item{\code{initialize(m)}:}{ Initializes the object. m should be a
      numeric matrix (use by calling new as in examples).
    }
    \item{\code{squeeze(target_dim, iter_no)}:}{ Squeezes points into
      \code{target_dim} dimensions in \code{iter_no} iterations. By
      default the function will continue iterating until no more stress
      is removed. This can be changed by \code{residual(FALSE)}.
      Returns a
      named list containing the new coordinates in \code{$pos}, the
      stress evolution in \code{$stress}, the mapping dimensionality in
      \code{$mapDims} and the stress (sum error) at each node in
      \code{$node_stress}.
    }
    \item{\code{squeezeDF(dimFactors)}:}{ Squeezes points using a
      reduction in dimensions as specifed by the numeric matrix
      \code{dimFactors}, which can be obtained from the
      \code{.dim.factors} methods. Returns a named list as for \code{squeeze}.
    }
    \item{\code{useOpenMP(useOMP)}:}{ Specify whether to use openMP
      based parallelisation. \code{useOMP} should be either \code{TRUE} or
      \code{FALSE}.
    }
    \item{\code{residual(remResidual)}:}{ Specify whether residual
      stress should be removed by further iterations after the specified
      number of iterations (the default state). \code{remResidual}
      should be \code{TRUE} or \code{FALSE}.
    }
  }
}

