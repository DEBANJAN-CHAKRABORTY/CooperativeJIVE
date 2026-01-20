library(Matrix)

datasets = list.dirs("DATASET",recursive=FALSE)
for (set in datasets){
  start_time <- Sys.time()
  print(set)
  modality = list.dirs(set,recursive=FALSE)
  Data<-list()
  for (view in seq(length(modality))){
    Data[[view]] = t(as.matrix(read.table(paste0(modality[view],"/clustering.txt"),sep=" ",header=FALSE)))
  }
  gt = as.vector(read.table(paste0(set,"/cluster_ground_truth.txt"),sep=" ",header=FALSE))
  modalities = c(list.dirs(set,full.names=FALSE,recursive=FALSE))
  K = length(unique(unlist(gt, use.names = FALSE)))
  n = length(unlist(gt,use.names = FALSE))
  samples = seq(1,n)
  LogData=Data
  M=length(Data)
  library(r.jive)
  Results = jive(Data,showProgress=FALSE)
  Joint = Results$joint
  Ind = Results$individual
  rankJ = Results$rankJ
  rankI = Results$rankA
  Ind_res = list()
  Indicator_res = list()
  joint = list()
  library(cluster)
  for (view in seq(M)){
    cat(rankI[view])
    A = data.frame(Ind[view])
    B = data.frame(Joint[view])
    C = as.matrix(A+B)
    temp = t(C)
    restest = kmeans(temp,K)$cluster
    res_indicator = model.matrix(~ as.factor(restest) - 1)
    dist_matrix = dist(temp)
    sil_info <- silhouette(restest, dist_matrix)
    summary(sil_info)
    mean_silhouette_score <- summary(sil_info)$avg.width
    cat(paste0("\t",modalities[view],"\t",mean_silhouette_score,"\n"))
    Ind_res[[view]] = restest
    Indicator_res = cbind(Indicator_res,res_indicator)
    joint = rbind(joint,Joint[view])
  }
  res_ind = do.call(rbind,Ind_res)
  jt = t(do.call(rbind,joint))
  dfjt = data.frame(jt)
  res_joint = kmeans(jt,K)$cluster
  dist_matrix = dist(jt)
  sil_info <- silhouette(res_joint, dist_matrix)
  mean_silhouette_score <- summary(sil_info)$avg.width
  cat(paste0(rankJ,"\tJoint\t",mean_silhouette_score,"\n"))
  result = rbind(res_joint,res_ind)
  df=data.frame(result)
  BPG = cbind(model.matrix(~ as.factor(res_joint) - 1),Indicator_res)
  B <- matrix(as.numeric(BPG), nrow = nrow(BPG), ncol = ncol(BPG))
  Nx <- nrow(B)
  Ny <- ncol(B)
  Dx <- diag(rowSums(B, na.rm = TRUE))
  Dy <- diag(colSums(B, na.rm = TRUE))
  Wy <- t(B)%*%solve(Dx)%*%B
  d <- rowSums(Wy)
  d[d == 0] <- .Machine$double.eps
  D_inv_sqrt <- sparseMatrix(i = 1:Ny, j = 1:Ny, x = 1/sqrt(d))
  nWy <- D_inv_sqrt %*% Wy %*% D_inv_sqrt
  nWy <- (nWy + t(nWy)) / 2
  decomp <- eigen(as.matrix(nWy))
  evec_raw <- decomp$vectors
  evals <- decomp$values
  idx <- order(evals, decreasing = TRUE)[1:K]
  Ncut_evec <- D_inv_sqrt %*% evec_raw[, idx]
  evec <- Dx %*% B %*% Ncut_evec
  evec <- as.matrix(evec)
  row_norms <- sqrt(rowSums(evec^2)) + 1e-10
  evec_norm <- sweep(evec, 1, row_norms, "/")
  km_res <- kmeans(evec_norm, K)
  label_final <-km_res$cluster
  end_time <- Sys.time()
  duration <- as.numeric(end_time - start_time, units = "secs")
  print(duration)
}