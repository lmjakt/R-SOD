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

R_DimSqueezer\cr
Uses openMP to parallelise the mapping process\cr

R_CL_DimSqueezer\cr
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
The basic functions are accessed from DimSqueezer objects which are created using the new function by providing the positions of nodes in N-dimensional space. \code{ds <- new(R_DimSqueezer, data)} The resulting object can be used to repeat the mapping using different variations of the theme.
Functions are also provided to specify the manner in which dimensionality is reduced \code{parallel.dim.factors()}, \code{parallel.exp.dim.factors()}, and others.   

}
\author{
Martin Jakt

Maintainer: Martin Jakt <mjakt@z2.keio.jp>
}
\references{
~~ Literature or other references for background information ~~
}
%%~~ Optionally other standard keywords, one per line, from file KEYWORDS in ~~
%%~~ the R documentation directory ~~
\keyword{ package }
\seealso{
~~ Optional links to other man pages, e.g. ~~
~~ \code{\link[<pkg>:<pkg>-package]{<pkg>}} ~~
}
\examples{
%% ~~ simple examples of the most important functions ~~

data <- matrix(nrow=100, ncol=6)
for(i in 1:ncol(data)){
   data[,i] <- rnorm(100)
}

ds1 <- new(R_DimSqueezer, data)
sq1 <- ds1$squeeze(2, 1000)

ds2 <- new(R_CL_DimSqueezer, data)
sq2 <- ds2$squeeze(2, 1000, 64)

plot.points(sq1)
plot.stress(sq2)
}
