library(RSQLite)
library(ggplot2)
library(data.table)
library(reshape2)
library(plyr)

db <- dbConnect(SQLite(), ".bench.db")
dbq <- function(...) { dbGetQuery(db, ...) }
dt <- data.table(dbq("select * from results"))

## rename variables to agree with paper
dt$D <- dt$n
dt$n <- dt$lineages * dt$D
dt$D <- factor(dt$D)

dt$model[dt$model == 'chen'] <- 'Chen'
dt$model[dt$model == 'moran'] <- 'momi'

dt$initial <- factor(dt$site == 0)
levels(dt$initial) <- c("Per SNP", "Precomputation")

rename(dt, c('model'='method'))

dt.summary <- dt[, list(avg_t=mean(time),runs=length(time),negs=sum(result < 0,na.rm=TRUE),nans=sum(is.na(result))), by=list(method, n, D, initial)]

p <- ggplot(dt.summary, aes(x=n, y=avg_t, color=D)) + geom_line() + geom_point() + ylab("Seconds") + theme(legend.position="top") + facet_grid(initial ~ method, scales='free_y')
plot(p)
ggsave('figures/timing.pdf')

p <- ggplot(dt.summary, aes(x=n, y=avg_t, color=D, linetype=method)) + geom_line() + geom_point(aes(shape=method)) + ylab("Seconds") + theme(legend.position="top") + facet_grid(initial ~ .) + scale_y_log10(breaks=c(1e-4,1e-2,1,1e2,1e4)) + scale_x_log10(breaks=c(2,4,8,16,32,64,128,256))
plot(p)
ggsave('figures/timing_log.pdf')

write.table(format(dcast(subset(dt.summary, initial=='Per SNP'), formula=n+D~method, value.var='avg_t'),digits=4,scientific=T),
            file='figures/per_snp.txt', quote=F, row.names=F, sep=' & ', eol=' \\\\ \\hline \n')

write.table(format(dcast(subset(dt.summary, initial=='Precomputation'), formula=n+D~method, value.var='avg_t'),digits=4,scientific=T),
            file='figures/precomputation.txt', quote=F, row.names=F, sep=' & ', eol=' \\\\ \\hline \n')

#dt$transformed.result <- sign(dt$result) * log10(1 + abs(dt$result))

dt2 <- dcast(dt, n + D + site + run_id + state + tree + initial ~ method, value.var = "result")
dt2 <- dt2[sample(nrow(dt2)),]

library(scales)
sgn_log <- function(x) sign(x) * log10(1 + abs(x))
inv_sgn_log <- function(x) sign(x) * (10^(abs(x)) - 1)
sgn_log_trans <- function() trans_new("sgn_log", sgn_log, inv_sgn_log, trans_breaks(sgn_log, inv_sgn_log))
p <- ggplot(dt2, aes(x=momi,y=Chen,color=n)) + geom_abline(color='red', linetype='dashed') + geom_point() + scale_y_continuous(trans = sgn_log_trans()) + scale_x_continuous(trans = sgn_log_trans())
plot(p)
ggsave('figures/accuracy.pdf')

p <- ggplot(dt2, aes(x=momi,y=Chen,color=n)) + geom_abline(color='red', linetype='dashed') + geom_point() + scale_y_continuous(trans = sgn_log_trans()) + scale_x_continuous(trans = sgn_log_trans()) + facet_grid(D ~ n)
plot(p)

p <- ggplot(subset(dt2, n <= 64), aes(x=momi,y=Chen,color=n)) + geom_abline(color='red', linetype='dashed') + geom_point() + scale_y_continuous(trans = sgn_log_trans()) + scale_x_continuous(trans = sgn_log_trans()) + facet_grid(D ~ n)
plot(p)
