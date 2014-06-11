\name{SOD: hsv.scale}
\alias{hsv.scale}

\title{hsv.scale: A colour gradient}

\description{
  Provides colours for a vector of values, with colours ranging from
  blue (min) to purple (max) via cyan, green, yellow and
  red. The colours are specified via the hsv method.
}

\usage{
hsv.scale(v, sat=1, val=0.75, alpha=1, min.v=min(v), max.v=max(v))
}

\arguments{
  \item{v}{A numeric vector}
  \item{sat}{The saturation of the resulting colors in the range 0-1. A
    vector or a single value. The values will be recycled for individual
    colors.}
  \item{val}{The value (i.e. how bright) of the resulting colors. Will
    be recycled as for sat.}
  \item{alpha}{The alpha values of the resulting colors, recycled as for
  sat and val.}
  \item{min.v}{The logical minimum of the range of values (to allow
  comparisons between different vectors). min.v should be equal to or
  lower than the minimum value in the range, but this is not enforced}
  \item{max.v}{The logical maximum of the range of values.}
}

\value{
  A vector of colors, one each for the elements in \code{v}.\cr
  The colors are scaled from blue (lowest) to purple (highest)
  via cyan, green, yellow and red.
}

\examples{
v <- 1:30
plot(v, rep(1, length(v)), col=hsv.scale(v), pch=19, cex=3)

## set the saturation to 0.5
plot(v, rep(1, length(v)), col=hsv.scale(v, sat=0.5), pch=19, cex=3)

## cycle saturation values
plot(v, rep(1, length(v)), col=hsv.scale(v, sat=c(0.3,0.6,0.9)), pch=19, cex=3)

## set the value to 0.4 (dark colours)
plot(v, rep(1, length(v)), col=hsv.scale(v, val=0.5), pch=19, cex=3)

## the colours are not related to the position:
v2 <- sample(v, length(v))
plot(v, rep(1, length(v)), col=hsv.scale(v2), pch=19, cex=3)

}
