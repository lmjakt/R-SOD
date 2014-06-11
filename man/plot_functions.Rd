\name{SOD.plot.functions}
\alias{plt.points}
\alias{plt.concentric}
\alias{plt.stress}

\title{SOD plotting functions}
\description{
  Functions for visualising points mapped by the SOD algorithm,
  \itemize{
    \item
    \code{plt.points} plots mapped points
    \item
    \code{plt.concentric} fancy plots of points
    \item
    \code{plt.stress} plots the change of the sum of errors in the
    inter-node distances.
  }
  These are simple convenience functions that the user may wish to
  extend depending on the visualisation requirements.

}

\usage{
plt.points(sq, col=hsv.scale(sq$node_stress), x=1, y=2,
invert.y=FALSE, xlab=NA, ylab=NA, ...)

plt.concentric(sq, cex.data, col=hsv.scale(1:ncol(cex.data)),x=1,
y=2, cex.max=3, invert.y=FALSE, pch=19, xlab=NA, ylab=NA, leg.pos=NULL, ...)

plt.stress(sq, bg.alpha=0.5, bg.sat=1, bg.val=0.75, col='black', lwd=1,
lty=1, main="Error / Dimension",
xlab="iteration", ylab="error / dimensionality")

}

\arguments{
  \item{sq}{A named list produced by the squeeze functions containing
    'stress', 'mapDims', 'pos', 'time', 'node_stress'.}
  \item{col}{A vector of colours for each row of sq$pos, or for
    plt.concentric a colour for each column of cex.data. For
  plt.stress the \code{col} argument gives the colour for the stress points.}
  \item{x}{The dimension to plot on the x-axis.}
  \item{y}{The dimension to plot on the y-axis.}
  \item{invert.y}{A boolean determining whether the y-axis should be inverted}
  \item{pch}{The plot character to use. Defaults to 19 (solid circles)
    for plt.concentric.}
  \item{cex.max}{The size of the points representing maximal values for
    plt.concentric.}
  \item{cex.data}{A numeric matrix providing numerical descriptions of
    the items represented by each point (1 row per point). Each column
    represents a numerical descriptor of some sort, and will be
    plotted as a series of conentric points where the incremental
    area of each point is proportional to the value scaled across all
    columns}
  \item{leg.pos}{The position of an optional legend. If NULL (the
    default) no legend will be drawn. The leg.pos should be specified as
    for \code{legend}, (eg. 'topright', 'bottomleft'). The colnames of
    cex.data will be used for the legend text.}
  \item{bg.alpha, bg.sat, bg.val}{Parameters specifying the background
    colours for the plt.stress functions. These colours are used to
    visualise the change in dimensionality through the mapping process.}
  \item{lwd, lty}{The width and type of line used to plot stress levels
    in plt.stress}
  \item{main}{The title for the plot produced by plt.stress. Defaults
    to "Error / Dimension"}
  \item{xlab}{The label of the x-axis. Defaults to NA for point plots.}
  \item{ylab}{The label of the y-axis. Defaults to NA for point plots.}
  \item{...}{Other parameters to the plot function}
}

\value{
  Does not return anything.
}

\seealso{
  \code{\link{SOD-package}}
}

\examples{
## please see examples in the SOD documentation package
## ?SOD
##
}
