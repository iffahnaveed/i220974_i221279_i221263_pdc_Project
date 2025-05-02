#include <iostream>
#include <vector>
#include <unordered_set>
#include <queue>
#include <fstream>
#include <sstream>
#include <tuple>
#include <algorithm>
#include <climits>
#include <ctime>  // For measuring execution time

using namespace std;

const int INF = INT_MAX;

// Dijkstra's Algorithm for Shortest Path
pair<vector<int>, vector<int>> dijkstra(const vector<vector<pair<int, int>>>& adj, int start) {
    int n = adj.size() - 1; // Since nodes are 1-based
    vector<int> dist(n + 1, INF);
    vector<int> parent(n + 1, -1);
    dist[start] = 0;

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, start});

    while (!pq.empty()) {
        int u = pq.top().second;
        int current_dist = pq.top().first;
        pq.pop();

        if (current_dist > dist[u]) continue;

        for (const auto& edge : adj[u]) {
            int v = edge.first;
            int weight = edge.second;

            if (dist[v] > dist[u] + weight) {
                dist[v] = dist[u] + weight;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }

    return {dist, parent};
}

void readChangesFromFile(const string& filename, vector<tuple<char, int, int, int>>& edgeChanges) {
    ifstream infile(filename);
    if (!infile) {
        cerr << "Error opening changes file: " << filename << endl;
        return;
    }

    string line;
    while (getline(infile, line)) {
        if (line.empty() || line[0] == '%') continue;
        stringstream ss(line);
        char change_type;
        int u, v, weight;
        if (ss >> change_type >> u >> v >> weight) {
            edgeChanges.emplace_back(change_type, u, v, weight);
        }
    }
}

void displayShortestPaths(int start_node, int nrows, const vector<int>& dist, const vector<int>& parent, ostream& out) {
    out << "\nShortest paths from node " << start_node << ":\n";
    for (int i = 1; i <= nrows; ++i) {
        if (i == start_node) continue;
        
        out << "To node " << i << ": ";
        if (dist[i] == INF) {
            out << "Unreachable\n";
        } else {
            out << "Distance = " << dist[i] << ", Path = ";
            vector<int> path;
            for (int v = i; v != -1; v = parent[v]) 
                path.push_back(v);
            reverse(path.begin(), path.end());
            for (size_t j = 0; j < path.size(); ++j)
                out << path[j] << (j + 1 < path.size() ? " -> " : "\n");
        }
    }
}

void displayChanges(const vector<tuple<char, int, int, int>>& edgeChanges, ostream& out) {
    out << "\nApplied changes:\n";
    for (const auto& change : edgeChanges) {
        char type;
        int u, v, w;
        tie(type, u, v, w) = change;
        out << (type == 'D' ? "Deleted" : "Inserted") 
             << " edge: " << u << " -> " << v 
             << " (weight: " << w << ")\n";
    }
}

void displayAffectedNodes(const unordered_set<int>& affected, ostream& out) {
    if (!affected.empty()) {
        out << "\nNodes directly affected by changes: ";
        for (int node : affected) {
            out << node << " ";
        }
        out << endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph.mtx>\n";
        return 1;
    }

    // Open output file
    ofstream outfile("output.txt");
    if (!outfile) {
        cerr << "Error opening output file for writing\n";
        return 1;
    }

    // Read graph file
    ifstream infile(argv[1]);
    if (!infile) {
        cerr << "Cannot open graph file: " << argv[1] << endl;
        return 1;
    }

    // Read MatrixMarket header
    string line;
    int nrows = 0, ncols = 0, nentries = 0;
    while (getline(infile, line)) {
        if (line[0] == '%') continue;
        istringstream iss(line);
        if (!(iss >> nrows >> ncols >> nentries)) {
            cerr << "Error reading graph dimensions" << endl;
            return 1;
        }
        break;
    }

    // Read edges with weights (directed graph)
    vector<vector<pair<int, int>>> adj(nrows + 1);  // 1-based indexing
    int u, v, w;
    while (infile >> u >> v >> w) {
        if (u < 1 || u > nrows || v < 1 || v > nrows) {
            cerr << "Invalid node index: " << u << " or " << v << endl;
            continue;
        }
        adj[u].emplace_back(v, w);
    }

    int start_node = 1; // Set the desired start node here

// Initial SSSP computation
outfile << "Initial SSSP from node " << start_node << ":\n";
clock_t start_time = clock();  // Start measuring time for initial SSSP

auto [dist, parent] = dijkstra(adj, start_node);
displayShortestPaths(start_node, nrows, dist, parent, outfile);

clock_t end_time = clock();  // End measuring time for initial SSSP
double execution_time = double(end_time - start_time) / CLOCKS_PER_SEC;
outfile << "Execution time for initial SSSP: " << execution_time << " seconds\n";

// Read and process edge changes
vector<tuple<char, int, int, int>> edgeChanges;
readChangesFromFile("changes.txt", edgeChanges);
displayChanges(edgeChanges, outfile);

// Process changes and track affected nodes
unordered_set<int> affected;
for (const auto& change : edgeChanges) {
    char type;
    int u, v, w;
    tie(type, u, v, w) = change;

    if (type == 'D') {
        auto& edges = adj[u];
        auto it = remove_if(edges.begin(), edges.end(),
                            [v](const pair<int, int>& p) { return p.first == v; });
        if (it != edges.end()) {
            edges.erase(it, edges.end());
            affected.insert(u);
            affected.insert(v);
            outfile << "Successfully deleted edge: " << u << " -> " << v << endl;
        } else {
            outfile << "Edge not found for deletion: " << u << " -> " << v << endl;
        }
    } else if (type == 'I') {
        adj[u].emplace_back(v, w);
        affected.insert(u);
        affected.insert(v);
        outfile << "Successfully inserted edge: " << u << " -> " << v << " (weight: " << w << ")" << endl;
    }
}
displayAffectedNodes(affected, outfile);

// Recompute and display updated paths from the same node
outfile << "\nUpdated SSSP from node " << start_node << " after changes:\n";
start_time = clock();

tie(dist, parent) = dijkstra(adj, start_node);

outfile << "\n========================================\n";
outfile << "Paths from node " << start_node << " (updated):\n";
outfile << "========================================\n";

for (int i = 1; i <= nrows; ++i) {
    if (i == start_node) continue;

    outfile << "To node " << i << ": ";
    if (dist[i] == INF) {
        outfile << "Unreachable";
        if (affected.count(i)) outfile << " [AFFECTED]";
        outfile << "\n";
    } else {
        outfile << "Distance = " << dist[i];
        if (affected.count(i)) outfile << " [AFFECTED]";
        outfile << ", Path = ";

        vector<int> path;
        for (int v = i; v != -1; v = parent[v])
            path.push_back(v);
        reverse(path.begin(), path.end());

        for (size_t j = 0; j < path.size(); ++j) {
            if (affected.count(path[j])) {
                outfile << "[" << path[j] << "]";
            } else {
                outfile << path[j];
            }
            outfile << (j + 1 < path.size() ? " -> " : "\n");
        }
    }
}

end_time = clock();
execution_time = double(end_time - start_time) / CLOCKS_PER_SEC;
outfile << "Execution time for updated SSSP: " << execution_time << " seconds\n";


    return 0;
}
