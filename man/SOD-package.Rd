\name{SOD-package}
\alias{SOD-package}
\alias{SOD}
\docType{package}
\title{
Heuristict mapping of N-dimensional nodes to M-dimensions
}
\description{
Maps points in N-dimensions to M-dimensions using a heuristic that 
tries to retain correct inter-node distances by rearranging nodes
using errors in the inter-node distances induced by gradually shrinking
the number of dimensions.

The primary purpose of this is is to represent N-dimensional nodes 
(points) in a 2-D or 3-D space in order to visualise the relationships
between nodes.

Since the method is computationally intensive, two classes are provided\cr

DimSqueezer\cr
Uses openMP to parallelise the mapping process\cr

DimSqueezer_CL\cr
Uses openCL to perform the mapping operation on a GPU
}
\details{
\tabular{ll}{
Package: \tab SOD\cr
Type: \tab Package\cr
Version: \tab 1.0\cr
Date: \tab 2013-10-10\cr
License: \tab GPL-2\cr
}
The basic functions are accessed from DimSqueezer objects which are
created using the new function by providing the positions of nodes in
N-dimensional space:\cr \code{ds <- new(DimSqueezer, data)}.\cr
The resulting object can be used to repeat the mapping using different
parameter combinations.\cr
Functions are also provided to specify the manner in which
dimensionality is reduced\cr \code{parallel.dim.factors()}\cr
\code{parallel.exp.dim.factors()}, and how to plot point positions and
stress evolution.   

}
\author{
Martin Jakt

Maintainer: Martin Jakt <mjakt@z2.keio.jp>
}

\references{
  \url{http://martin.jakt.org.uk/r-sod/}\cr
  \url{http://www.gitorious.org/r-sod/}
}
%%~~ Optionally other standard keywords, one per line, from file KEYWORDS in ~~
%%~~ the R documentation directory ~~
\keyword{ package }
\seealso{
\code{\link{DimSqueezer}}\cr 
\code{\link{DimSqueezer_CL}}\cr 
\code{\link{parallel.dim.factors}}\cr 
\code{\link{parallel.exp.dim.factors}}
}

\examples{
%% ~~ simple examples of the most important functions ~~

require(SOD)

## read in the f186 data set
data("f186", package="SOD")

## f186 contains information about a set of cells. We
## want to use columns 7:12, which contain transcript
## densities to visualise the relationships between the
## cells.

## Make a normal (openMP) accelarated squeezer object

## use a subset of the rows to reduce the computation time
## for automated checks

r <- 1:200
ds1 <- new(DimSqueezer, as.matrix(f186[r,7:12]) )

## default squeeze, 1000 iterations resulting in a
## 2 dimensional object.
sq1 <- ds1$squeeze(2, 200)

\dontrun{
## Make an openCL accelerated squeezer object
## note this is only available where both the physical
## hardware and the appropriate development enviroment
## is available.
ds2 <- new(DimSqueezer_CL, as.matrix(f186[,7:12]) )

## squeeze, 1000 iterations using a work group size of 64
## giving a 2 dimensional representation.
sq2 <- ds2$squeeze(2, 200, 64)
}
## functions to plot the relationships
plt.points(sq1)

## use columns of f186 to make a more interesting
## plot
plt.points(sq1, col=hsv.scale(f186[r,'Etv2']), pch=19)
plt.points(sq1, col=hsv.scale(f186[r,'p_Etv2']), pch=19)

## and a funky plot
plt.concentric(sq1, f186[r,7:12], cex.max=5, pch=19, leg.pos='topleft')

## and to plot the stress mapping.
plt.stress(sq1)
}
