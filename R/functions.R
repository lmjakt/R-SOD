##
## A set of functions that are useful for visualsing squeezed data ##
loadModule("mod_R_DimSqueezer", TRUE)

## makes a color for each of level of v 
hsv.scale <- function(v, sat=1, val=0.75, alpha=1.0){
    cols <- v - min(v);
    max.c <- max(cols)
    ## hue of 0.6665 = blue
    ## hue of 1 = 0 = red
    ## hue of 0.8331 = purple
    
    ## a reasonable range may run from blue -> purple, but avoiding the purple -> red transition
    ## giving us a total rango of
    
    ## blue -> red (0.6665), red -> purple (1 - 0.8331 = 0.1669 )
    ## total range  of 0.8334
    
    ## consider full circle as 10,000 then we can simply do something like
    cols <- 8334 * (cols/max.c);
    cols <- 6665 - cols
    cols[ cols < 0 ] <- 10000 + cols[ cols < 0 ] ## very ugly.
    cols <- cols / 10000
    cols.v <- vector(length=length(cols))
    for( i in 1:length(cols))
        cols.v[i] <- hsv( cols[i], sat, val, alpha )
    cols.v
}

## sq squeezed data
plot.points <- function(sq, col=hsv.scale(sq$node_stress), x=1, y=2, cex=1,
                        invert.y=FALSE, pch=1, xlab=NA, ylab=NA, ...){
    xv = sq$pos[,x]
    yv = sq$pos[,y]
    if(invert.y)
        yv = -yv
    plot(xv, yv, col=col, cex=cex, bg=col, pch=pch, xlab=xlab, ylab=ylab, ...)
}

plot.concentric <- function(sq, cex.data, cols=hsv.scale(1:ncol(cex.data)), x=1, y=2, cex.max=3, invert.y=FALSE, pch=1, xlab=NA, ylab=NA){
    xv <- sq$pos[,x]
    yv <- sq$pos[,y]
    if(invert.y)
        yv <- -yv
    p.cex <- matrix(nrow=nrow(cex.data), ncol=ncol(cex.data), data=0)
    p.cex[,ncol(cex.data)] <- sqrt(cex.data[,ncol(cex.data)])
    for(i in (ncol(cex.data)-1):1){
        p.cex[,i] <- p.cex[,(i+1)] + sqrt(cex.data[,i])
    }
    ## scale to cex.max
    p.cex <- cex.max * p.cex / max(p.cex)
    plot(xv, yv, type='n', xlab=xlab, ylab=ylab)
    for(i in 1:ncol(p.cex))
        points(xv, yv, cex=p.cex[,i], col=cols[i], bg=cols[i], pch=pch)
}

plot.stress <- function(sq, bg.alpha=0.5, bg.sat=1, bg.val=0.75,
                        col='black', lwd=1, lty=1, main="Error / Dimension",
                        xlab="iteration", ylab="error / dimensionality"){
    bg.cols <- hsv.scale(1:ncol(sq$mapDims), alpha=bg.alpha, sat=bg.sat, val=bg.val)
    x.pts <- 1:length(sq$stress)
    max.x=length(sq$stress)
    plot(x.pts, sq$stress, type='n', xlab=xlab,
         ylab=ylab, main=main)
    y.range <- range(sq$stress)
    y.span <- y.range[2] - y.range[1]
    max.dim <- sum(sq$mapDims[1,])
    ## draw the background to indicate dimensionality
    for(i in ncol(sq$mapDims):2){
        d <- apply(sq$mapDims[, i:1], 1, sum)
        d <- y.range[1] + (d/max.dim)*y.span
        polygon( c(1, x.pts, max.x), c(y.range[1], d, y.range[1]), col=bg.cols[i], border=NA )
        
    }
    d <- sq$mapDims[,1]
    d <- y.range[1] + (d/max.dim) * y.span
    polygon( c(1, x.pts, max.x), c(y.range[1], d, y.range[1]), col=bg.cols[1], border=NA )
    points(x.pts, sq$stress, type='l', lwd=lwd, col=col, lty=lty)
}

parallel.dim.factors <- function(dim, iteration.no, red.end=iteration.no*0.75, target.dim=2){
    dimFactors <- matrix(nrow=iteration.no, ncol=dim, data=1.0)
    dimFactors[1:red.end, (target.dim+1):ncol(dimFactors) ] <- seq(from=1.0, to=0.0, length.out=red.end)
    dimFactors[(red.end+1):nrow(dimFactors), (target.dim+1):ncol(dimFactors) ] <- 0
    dimFactors
}

parallel.exp.dim.factors <- function(dim, iteration.no, target.dim=2, red.end=iteration.no * 0.9){
    dimFactors <- matrix(nrow=iteration.no, ncol=dim, data=1.0)
    dimFactors[1:red.end, (target.dim+1):ncol(dimFactors) ] <- 2^( -seq(from=0, to=10, length.out=(red.end)) )
    dimFactors[(red.end+1):nrow(dimFactors), (target.dim+1):ncol(dimFactors) ] <- 0
    dimFactors
}

serial.dim.factors <- function(dim, iteration.no, red.end=iteration.no*0.75, target.dim=2){
    dimFactors <- matrix(nrow=iteration.no, ncol=dim, data=1.0)
    d.l <- as.integer(red.end / (dim - target.dim))
    red.i <- 1
    for(i in (dim):(target.dim+1)){
        dimFactors[red.i:(red.i+d.l-1), i] <- seq(from=1.0, to=0, length.out=d.l)
        dimFactors[(red.i+d.l):nrow(dimFactors), i] <- 0
        red.i <- red.i + d.l
    }
    dimFactors
}


