from salsa.externals.karmadaInfo import KarmadaInfo


karmadaInfo = KarmadaInfo("/home/leonkiss/.kube/karmada.config")
resources = karmadaInfo.get_cluster_resource_util("member1")
print(resources)
deployments = karmadaInfo.get_microservice_replication("member1", ["app1", "app2", "app3"])
print("member1 deployments:\n" + str(deployments))
deployments = karmadaInfo.get_microservice_replication("member2", ["app2", "app3"])
print("member2 deployments:\n" + str(deployments))
