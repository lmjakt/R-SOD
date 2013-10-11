##
## A set of functions that are useful for visualsing squeezed data ##

## makes a color for each of level of v 
hsv.scale <- function(v, sat=1, val=0.75){
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
        cols.v[i] <- hsv( cols[i], sat, val )
    cols.v
}

## sq squeezed data
plot.points <- function(sq, col=hsv.scale(sq$node_stress), x=1, y=2, cex=1, invert.y=FALSE, pch=1, xlab=NA, ylab=NA){
    xv = sq$pos[,x]
    yv = sq$pos[,y]
    if(invert.y)
        yv = -yv
    plot(xv, yv, col=col, cex=cex, bg=col, pch=pch, xlab=xlab, ylab=ylab)
}
