#include <mpi.h>
#include <metis.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <climits>
#include <queue>
#include <unordered_set>
#include <tuple>
#include <algorithm>
#include <functional>

using namespace std;
const int INF = INT_MAX;

// Structure for MPI communication of node-distance pairs
struct NodeDistance { int node, distance; };
void convertToMetisFormat(const vector<vector<pair<int, int>>>& graph,
                          vector<idx_t>& xadj, vector<idx_t>& adjncy) {
    int n = graph.size();
    xadj.resize(n + 1);
    vector<idx_t> temp_adjncy;

    idx_t edge_count = 0;
    for (int i = 0; i < n; ++i) {
        xadj[i] = edge_count;
        for (auto& [neighbor, weight] : graph[i]) {
            temp_adjncy.push_back(neighbor);
            ++edge_count;
        }
    }
    xadj[n] = edge_count;
    adjncy = move(temp_adjncy);
}

void partitionGraph(const vector<vector<pair<int,int>>>& graph, int num_nodes, int nparts) {
    vector<idx_t> xadj, adjncy;
    convertToMetisFormat(graph, xadj, adjncy);

    idx_t nvtxs = num_nodes;
    idx_t ncon = 1;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options); // optional customization

    vector<idx_t> part(num_nodes); // output partition
    idx_t objval;

    int status = METIS_PartGraphKway(
        &nvtxs, &ncon, xadj.data(), adjncy.data(),
        NULL, NULL, NULL, &nparts,
        NULL, NULL, options, &objval, part.data());

    if (status == METIS_OK) {
        for (int i = 0; i < num_nodes; ++i) {
            //cout << "Node " << i + 1 << " is in partition " << part[i] << endl;
        }
    } else {
        cerr << "METIS partitioning failed!" << endl;
    }
}
// Read graph in Matrix Market format with weights
void readGraph(const string& filename,
               vector<vector<pair<int,int>>>& graph,
               int& num_nodes) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;
        break;
    }
    stringstream ss(line);
    int rows, cols, edges;
    ss >> rows >> cols >> edges;
    num_nodes = rows;
    graph.assign(num_nodes, {});
    int u, v, w;
    while (file >> u >> v >> w) {
        --u; --v;
        graph[u].emplace_back(v, w);
        graph[v].emplace_back(u, w); // undirected
    }
}

// Read edge changes from file
void readChanges(const string& filename,
                 vector<tuple<char,int,int,int>>& changes) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening changes file: " << filename << endl;
        return;
    }
    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;
        stringstream ss(line);
        char type;
        int u, v, w;
        if (ss >> type >> u >> v >> w) {
            changes.emplace_back(type, u-1, v-1, w);
        }
    }
}

// Parallel Dijkstra with MPI: uses partition ownership and seed nodes
void parallelDijkstra(
    const vector<vector<pair<int,int>>>& graph,
    vector<int>& dist,
    vector<int>& parent,
    const vector<int>& local_nodes,
    int rank, int size,
    const vector<int>& seed_nodes
) {
    int N = graph.size();
    vector<bool> visited(N, false);
    auto cmp = [](const NodeDistance &a, const NodeDistance &b) {
        return a.distance > b.distance;
    };
    priority_queue<NodeDistance, vector<NodeDistance>, decltype(cmp)> pq(cmp);
    // Seed priority queue
  	for (int u : seed_nodes) {
      if (dist[u] < INF && !visited[u]) {
        pq.push({u, dist[u]});
      }
    }

    while (true) {
        NodeDistance local_min = {-1, INF};
        if (!pq.empty()) {
            NodeDistance top = pq.top(); pq.pop();
            if (top.distance <= dist[top.node]) local_min = top;
        }
        // Gather local minima
        vector<NodeDistance> allmins;
        if (rank == 0) allmins.resize(size);
        MPI_Gather(&local_min, sizeof(NodeDistance), MPI_BYTE,
                   allmins.data(), sizeof(NodeDistance), MPI_BYTE,
                   0, MPI_COMM_WORLD);
        // Rank 0 selects global minimum
        NodeDistance global_min = {-1, INF};
        if (rank == 0) {
            for (auto &nd : allmins) {
                if (nd.node >= 0 && nd.distance < global_min.distance)
                    global_min = nd;
            }
        }
        MPI_Bcast(&global_min, sizeof(NodeDistance), MPI_BYTE, 0, MPI_COMM_WORLD);
        if (global_min.node < 0 || global_min.distance == INF) break;
        int u = global_min.node;
        visited[u] = true;
        bool i_own = find(local_nodes.begin(), local_nodes.end(), u) != local_nodes.end();
        if (i_own) {
            for (auto &e : graph[u]) {
                int v = e.first, w = e.second;
                if (!visited[v] && dist[v] > dist[u] + w) {
                    dist[v] = dist[u] + w;
                    parent[v] = u;
                    if (find(local_nodes.begin(), local_nodes.end(), v) != local_nodes.end())
                        pq.push({v, dist[v]});
                }
            }
        }
        // Broadcast updated arrays
        MPI_Bcast(dist.data(), N, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(parent.data(), N, MPI_INT, 0, MPI_COMM_WORLD);
        vector<int> vis_i(N);
        if (rank == 0) for (int i = 0; i < N; ++i) vis_i[i] = visited[i];
        MPI_Bcast(vis_i.data(), N, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) for (int i = 0; i < N; ++i) visited[i] = vis_i[i];
    }
}

// Process edge changes & trigger incremental update
void processChangesMPI(
    const vector<tuple<char,int,int,int>>& changes,
    vector<vector<pair<int,int>>>& graph,
    vector<int>& dist,
    vector<int>& parent,
    unordered_set<int>& affected,
    const vector<int>& part,
    int rank, int size
) {
    int N = graph.size();
    // Build local_nodes
    vector<int> local_nodes;
    for (int i = 0; i < N; ++i)
        if (part[i] == rank) local_nodes.push_back(i);
    // Handle deletions
    for (auto &c : changes) {
        char t; int u,v,w;
        tie(t,u,v,w) = c;
        if (t == 'D' && part[u] == rank) {
            auto& edges = graph[u];
            edges.erase(remove_if(edges.begin(), edges.end(),
                        [v](auto &p){return p.first==v;}), edges.end());
        }
    }
    // DFS to find unreachable
    vector<int> recompute;
    if (rank == 0) {
        vector<bool> via(N,false);
        function<void(int)> dfs = [&](int u) {
            via[u] = true;
            for (auto &e : graph[u]) if (!via[e.first] && parent[e.first]==u)
                dfs(e.first);
        };
        dfs(0);
        for (int i = 0; i < N; ++i) {
            if (dist[i] < INF && !via[i]) {
                dist[i] = INF;
                parent[i] = -1;
                recompute.push_back(i);
                affected.insert(i);
            }
        }
    }
    int m = recompute.size();
    MPI_Bcast(&m,1,MPI_INT,0,MPI_COMM_WORLD);
    recompute.resize(m);
    MPI_Bcast(recompute.data(), m, MPI_INT, 0, MPI_COMM_WORLD);
    // Handle insertions
    for (auto &c : changes) {
        char t; int u,v,w;
        tie(t,u,v,w) = c;
        if (t == 'I') {
            if (part[u] == rank) graph[u].emplace_back(v,w);
            if (dist[u] < INF && (dist[v]==INF || dist[v] > dist[u]+w)) {
                dist[v] = dist[u]+w;
                parent[v] = u;
                recompute.push_back(v);
                affected.insert(v);
            }
        }
    }
    // Run parallel Dijkstra seeded on recompute
    parallelDijkstra(graph, dist, parent,
                     local_nodes, rank, size,
                     recompute);
}

// Output routines
void displayResults(int src, int N,
                    const vector<int>& dist,
                    const vector<int>& parent,
                    const unordered_set<int>& affected,
                    ofstream &out) {
    out << "\n========================================\n";
    out << "Paths from node " << src+1 << " (updated):\n";
    out << "========================================\n";
    for (int i = 0; i < N; ++i) {
        if (i == src) continue;
        out << "To node " << i+1 << ": ";
        if (dist[i] == INF) {
            out << "Unreachable";
            if (affected.count(i)) out << " [AFFECTED]";
            out << "\n";
        } else {
            out << "Distance = " << dist[i];
            if (affected.count(i)) out << " [AFFECTED]";
            out << ", Path = ";
            vector<int> path;
            for (int v=i; v!=-1; v=parent[v]) path.push_back(v);
            reverse(path.begin(), path.end());
            for (int j=0; j<path.size(); ++j) {
                if (affected.count(path[j])) out << "[" << path[j]+1 << "]";
                else out << path[j]+1;
                out << (j+1<path.size()?" -> ":"\n");
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) 
            cerr << "Usage: " << argv[0] << " <graph.mtx>\n";
        MPI_Finalize();
        return 1;
    }

    // ─── 1) Read & partition the graph on rank 0 ───────────────────────────────
    vector<vector<pair<int,int>>> graph;
    vector<int> part;
    int N = 0;
     readGraph(argv[1], graph, N);

   // 2) Only rank 0 partitions it
   if (rank == 0) {
       part.resize(N);
       partitionGraph(graph, N, size);
   }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        graph.resize(N);
        part.resize(N);
    }
    
    MPI_Bcast(part.data(), N, MPI_INT, 0, MPI_COMM_WORLD);

    // ─── 2) Build local_nodes list ──────────────────────────────────────────────
    vector<int> local_nodes;
    for (int i = 0; i < N; ++i)
        if (part[i] == rank)
            local_nodes.push_back(i);

    // ─── 3) Initialize dist & parent ────────────────────────────────────────────
    vector<int> dist(N, INF), parent(N, -1);
    int src = 0;
    if (find(local_nodes.begin(), local_nodes.end(), src) != local_nodes.end())
        dist[src] = 0;

    // ─── 4) Run the parallel Dijkstra ───────────────────────────────────────────
    double t0 = MPI_Wtime();
    parallelDijkstra(graph, dist, parent,
                 /*local_nodes=*/local_nodes,
                 rank, size,
                 /*seed_nodes=*/std::vector<int>{src});

    double init_time = MPI_Wtime() - t0;

    // ─── 5) Gather the *global* dist & parent on rank 0 ────────────────────────
    vector<int> global_dist, global_parent;
    if (rank == 0) {
        global_dist.resize(N);
        global_parent.resize(N);
    }
    MPI_Reduce(dist.data(), 
               rank==0 ? global_dist.data() : nullptr,
               N, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(parent.data(),
               rank==0 ? global_parent.data() : nullptr,
               N, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    // Overwrite local arrays on rank0 so displayResults sees the true global state
    if (rank == 0) {
        dist = move(global_dist);
        parent = move(global_parent);
    }

    // ─── 6) On rank 0: print initial SSSP ──────────────────────────────────────
    if (rank == 0) {
        ofstream out("output_mpi.txt");
        out << "Execution time for initial SSSP: " << init_time << " seconds\n";
        cout << "Execution time for initial SSSP: " << init_time << " seconds\n";
        out << "\nInitial SSSP Results:\n";
        displayResults(src, N, dist, parent, {}, out);
        out.close();
    }

    // ─── 7) Read & broadcast edge changes ───────────────────────────────────────
    vector<tuple<char,int,int,int>> changes;
    if (rank == 0) readChanges("changes.txt", changes);
    int C = changes.size();
    MPI_Bcast(&C, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) changes.resize(C);
    for (int i = 0; i < C; ++i) {
        char t; int u,v,w;
        if (rank == 0) tie(t,u,v,w) = changes[i];
        MPI_Bcast(&t, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(&u, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) changes[i] = make_tuple(t,u,v,w);
    }

    // ─── 8) Apply changes in parallel and re-run Dijkstra updates ──────────────
    unordered_set<int> affected;
    t0 = MPI_Wtime();
    processChangesMPI(changes, graph, dist, parent, affected, part, rank, size);
    double upd_time = MPI_Wtime() - t0;

    // ─── 9) Gather the *updated* global dist & parent on rank 0 ───────────────
    if (rank == 0) {
        global_dist.assign(N, INF);
        global_parent.assign(N, -1);
    }
    MPI_Reduce(dist.data(), 
               rank==0 ? global_dist.data() : nullptr,
               N, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(parent.data(),
               rank==0 ? global_parent.data() : nullptr,
               N, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        dist = move(global_dist);
        parent = move(global_parent);
    }

    // ─── 10) On rank 0: append final results ───────────────────────────────────
    if (rank == 0) {
        ofstream out("output_mpi.txt", ios::app);
        out << "\nApplied changes:\n";
        for (auto &c : changes) {
            char t; int u,v,w;
            tie(t,u,v,w) = c;
            out << (t=='D' ? "Deleted" : "Inserted")
                << " edge: " << u+1 << " -> " << v+1
                << " (weight: " << w << ")\n";
        }
        if (!affected.empty()) {
            out << "\nNodes directly affected: ";
            for (int a : affected) out << a+1 << " ";
            out << "\n";
        }
        out << "\nFinal SSSP after changes:\n";
        displayResults(src, N, dist, parent, affected, out);
        out << "Execution time for incremental SSSP update: " << upd_time << " seconds\n";
        cout << "Execution time for incremental SSSP update: " << upd_time << " seconds\n";
        out.close();
    }

    MPI_Finalize();
    return 0;
}

