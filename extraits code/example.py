
def extraire_commentaires() -> None:
    for data_dir in list_data_dir:
        for stream_file_adr in os.listdir(os.path.join(path_data, data_dir)):
            if stream_file_adr[-3:] == "txt":
                auteur: str = stream_file_adr.split(" - ")[0]
                titre: str = " - ".join(stream_file_adr.split(" - ")[1:])[:-11]
                print(titre)
                date: str = auteur.split("]")[0][1:]
                auteur = auteur.split("]")[1]
                try:
                    comments: pd.DataFrame = pd.read_csv(os.path.join(path_data, data_dir, stream_file_adr),
                                                         encoding="utf-8", sep="UTC]", engine="python", header=None)
                    comments.loc[:, "date"] = comments.loc[:, 0].apply(lambda z: z.split()[0][1:].strip())
                    comments.loc[:, "time"] = comments.loc[:, 0].apply(lambda z: z.split()[1].strip())
                    comments.drop(0, axis=1, inplace=True)
                    comments.loc[:, "author"] = comments.loc[:, 1].apply(lambda z: z.split(":")[0].strip())
                    comments.loc[:, "message"] = comments.loc[:, 1].apply(lambda z: ":".join(z.split(":")[1:]).strip())
                    comments.drop(1, axis=1, inplace=True)
                    comments.loc[:, "streamer"] = auteur.strip()
                    print(comments.head(2))
                    comments.to_csv(
                        os.path.join(path_data, "dataframe_comments", f"{data_dir}__{auteur}__{date}__{titre}.csv"),
                        sep=";", encoding="utf-8", index=False)
                except pd.errors.EmptyDataError:
                    pass



def followers_par_auteur():
    total: int = len(os.listdir(os.path.join(path_data, "dataframe_comments")))
    for data_dir in list_data_dir:
        auteurs = sorted(list(set([adr.split("__")[1].strip()
                                   for adr in os.listdir(os.path.join(path_data, "dataframe_comments")) if
                                   data_dir in adr])))
        for auteur in auteurs:
            print(auteur)
            subset: List[str] = [adr for adr in os.listdir(os.path.join(path_data, "dataframe_comments"))
                                 if data_dir in adr and adr.split("__")[1].strip() == auteur]
            dataframe: pd.DataFrame = pd.concat([pd.read_csv(os.path.join(path_data, "dataframe_comments", adr),
                                                            sep=";", encoding="utf-8") for adr in subset], axis=0)
            dataframe = dataframe.loc[:, ["date", "author"]].dropna()
            dataframe = dataframe.groupby(['date'])['author'].apply(lambda x: ' '.join(x)).reset_index()
            dataframe.loc[:, "author"] = dataframe.loc[:, "author"].apply(
                lambda z: " ".join(sorted(list(set(z.split())))))
            dataframe.loc[:, "streamer"] = auteur
            dataframe.loc[:, "event"] = data_dir
            dataframe.to_csv(os.path.join(path_data, "par_auteur_vw", f"{data_dir}_{auteur}.csv"), sep=";",
                             encoding="utf-8", index=False)
            total -= len(subset)
    print(total)


def compute_followers():
    all_data: pd.DataFrame = pd.concat([pd.read_csv(os.path.join(path_data, "par_auteur_vw", adr),
                                                    sep=";", encoding="utf-8", header=0) for adr
                                        in os.listdir(os.path.join(path_data, "par_auteur_vw"))],
                                       axis=0).drop("date", axis=1)
    for data_dir in list_data_dir:
        local_data: pd.DataFrame = all_data.loc[all_data.event == data_dir, :].drop("event", axis=1)
        local_data = local_data.groupby(['streamer'])['author'].apply(lambda x: ' '.join(x)).reset_index()
        local_data.loc[:, "author"] = local_data.loc[:, "author"].apply(
            lambda z: ' '.join(sorted(list(set(z.split())))))
        streamers = local_data.streamer.unique()
        local_data.set_index("streamer", inplace=True)
        matrix: pd.DataFrame = pd.DataFrame(index=streamers, columns=streamers)
        for streamer1 in streamers:
            for streamer2 in streamers:
                matrix.loc[streamer1, streamer2] = len(set(local_data.loc[streamer1, "author"].split())
                                                       .intersection(set(local_data.loc[streamer2, "author"].split())))
        print(matrix)

        print("***Réduction dimensionelle***")
        distance: pd.DataFrame = pd.DataFrame(index=streamers, columns=streamers)
        for streamer1 in streamers:
            for streamer2 in streamers:
                distance.loc[streamer1, streamer2] = 1 - 2 * matrix.loc[streamer1, streamer2] \
                                                     / (matrix.loc[streamer1, streamer1] + matrix.loc[
                    streamer2, streamer2])

        kmed: KMedoids = KMedoids(n_clusters=4)
        kmed.fit_predict(distance.values)
        cha: AgglomerativeClustering = AgglomerativeClustering(affinity="precomputed", linkage="complete")
        cha.fit_predict(distance.values)

        for cl, clustering in enumerate([kmed, cha]):
            mds: MDS = MDS(dissimilarity="precomputed")
            Xtr = mds.fit_transform(distance.values)
            plt.figure(figsize=(12, 12))
            plt.scatter(Xtr[:, 0], Xtr[:, 1], c=clustering.labels_)
            for i, nom in enumerate(distance.index):
                plt.annotate(text=nom, xy=(Xtr[i, 0] + 0.01, Xtr[i, 1]))
            plt.title(f"MDS sur followers - {data_dir}")
            plt.savefig(os.path.join(path_exports, "img", f"MDS-followers-{data_dir}.png"))

            tsne: TSNE = TSNE(n_components=2, learning_rate='auto', perplexity=Xtr.shape[0] // 3, metric="precomputed")
            Xtr = tsne.fit_transform(distance.values)
            plt.figure(figsize=(12, 12))
            plt.scatter(Xtr[:, 0], Xtr[:, 1], c=clustering.labels_)
            for i, nom in enumerate(distance.index):
                plt.annotate(text=nom, xy=(Xtr[i, 0] + 0.01, Xtr[i, 1]))
            plt.title(f"tSNE sur followers - {data_dir}")
            plt.savefig(os.path.join(path_exports, "img", f"tSNE-followers-{data_dir}--{cl}.png"))

        print("***Graphes****")

        threshold_edge = {"pixel_war": 800, "corpus_lausanne": 800}
        graphe_viewers: nx.Graph = nx.Graph()
        graphe_viewers.add_nodes_from([s for s in streamers
                                       if matrix.loc[matrix.loc[:, s] > threshold_edge[data_dir], :].shape[0] > 1])
        graphe_viewers.add_weighted_edges_from([(s1, s2, matrix.loc[s1, s2])
                                                for s1 in streamers for s2 in streamers if s1 != s2
                                                and matrix.loc[s1, s2] > threshold_edge[data_dir]])
        pos = nx.spring_layout(graphe_viewers)
        edge_trace: List[go.Scatter] = list()
        for edge in graphe_viewers.edges.data("weight"):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]
            edge_trace.append(go.Scatter(x=(x0, x1), y=(y0, y1), hoverinfo='none', mode='lines',
                                         line=dict(width=sqrt(weight) // 10, color="black")))
        node_x: List[float] = [pos[node][0] for node in graphe_viewers.nodes()]
        node_y: List[float] = [pos[node][1] for node in graphe_viewers.nodes()]
        node_labels = list(graphe_viewers.nodes())
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_labels,
                                marker=dict(
                                    color=cha.labels_,
                                    size=[sqrt(matrix.loc[l, l]) // 10 for i, l in enumerate(node_labels)]
                                ))
        fig = go.Figure(data=edge_trace + [node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.write_html(os.path.join(path_exports, "graphes", f"viewers-{data_dir}.html"))
        plt.close()
