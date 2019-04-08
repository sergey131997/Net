#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

const std::pair<double,double> radius = {15, 30};
const int max_dim_size = 1000;

class Generator {
public:

    Generator() {};

    Generator(std::string config) {
        srand(time(NULL));
        std::ifstream ifs(config);
        std::string tmp_string;
        std::getline(ifs, tmp_string);
        vector<int> tmp_val_vector;
        get_vals(tmp_string, tmp_val_vector);
        number_of_classes = tmp_val_vector[0];
        intersection.resize(number_of_classes);
        number_of_dimensions = tmp_val_vector[1];
        int type_of_data=0;
        while (std::getline(ifs, tmp_string)) {
            if (tmp_string == "intersect") {
                type_of_data = 0;
                continue;
            }
            if (tmp_string == "count") {
                type_of_data = 1;
                continue;
            }
            switch (type_of_data) {
                case 0:
                add_intersect(tmp_string);
                break;
                case 1:
                add_class_count(tmp_string);
                break;
            }
        }
        if (number_of_elements.size() != number_of_classes)
            throw std::runtime_error("Not enough elements in classess");
    }

    int generate() {
       data.clear();
       data.resize(number_of_classes);
       fill = true;
       std::vector<std::pair<int, int>> vals(number_of_classes);
       for(int i = 0; i < vals.size(); ++i)
           vals[i] = {intersection[i].size(), i};
       std::sort(vals.begin(), vals.end());
       std::reverse(vals.begin(), vals.end());
       std::vector<int> valid(number_of_classes, 0);
       int count = get_count_of_components();
       std::vector<double> direction(get_random_direction());
       std::vector<double> shift(number_of_dimensions, 0);
       not_valid.push_back(shift);
       for(std::pair<int,int>& idx : vals) {
           if (!valid[idx.second]) {
                dfs_fill(idx.second, shift, valid);
                next_shift(shift);
                not_valid.push_back(shift);
           }
       }
       return 0;
    }

    void print(std::string out_file) {
        std::ofstream ofs(out_file);
        ofs << "[\n";
        for (int i = 0; i < number_of_classes; ++i) {
            ofs << "[\n";
            for (int j = 0; j < data[i].size(); ++j) {
                ofs << "[";
                for(int k = 0; k < number_of_dimensions; ++k) {
                    ofs << data[i][j][k];
                    if (k + 1 < number_of_dimensions)
                        ofs << ", ";
                }
                ofs << "]";
                if (j + 1 < data[i].size())
                    ofs << ",";
                ofs << "\n";
            }
            ofs << "]";
            if (i + 1 < number_of_classes)
                ofs << ",";
            ofs << "\n";
        }
        ofs << "]";
    }

private:

    double get_distance(const std::vector<double>& a1,
                        const std::vector<double>& a2) {
        double dis = 0;
        for(int i = 0; i < number_of_dimensions; ++i)
            dis += std::pow(a1[i] - a2[i], 2);
        return std::sqrt(dis);
    }

    void dfs_fill(int v,
                  const std::vector<double>& pos,
                  std::vector<int>& valid,
                  const std::vector<double>& bad_dir = {0}) {
        valid[v] = 1;
        fill_vertex(v, pos);
        for(std::pair<int, double>& next : intersection[v])
            if (!valid[next.first]) {
                std::vector<double> dir = pos;
                bool flag = true;
                while(is_intersect(dir, true) || flag) {
                    dir=get_random_direction();
                    flag = false;
                    double distance = get_distance(next.second);
                    for(int i = 0; i < number_of_dimensions; ++i)
                        dir[i] = dir[i]*distance + pos[i];
                }
                dfs_fill(next.first, dir, valid);
            }
    }

    bool is_intersect(const std::vector<double>& a1, bool flag = false) {
        for(int i = 0; i < not_valid.size() - flag; ++i)
            if (get_distance(a1, not_valid[i]) < radius.second*2.5)
                return true;
        return false;
    }

    void fill_vertex(int v,
                     const std::vector<double>& pos) {
        std::vector<double> pos_vert(number_of_dimensions);
        for(int i =0; i< number_of_elements[v]; ++i) {
            pos_vert = get_random_direction();
            for(int i = 0; i < number_of_dimensions; ++i)
                pos_vert[i] = pos_vert[i] * (rand() % (int)(radius.first + rand() % (int)(radius.second - radius.first))) + pos[i];

            data[v].push_back(pos_vert);
        }
    }

    double get_distance(double inter_area) {
        return (1 - inter_area) * 2 * (radius.first + radius.second) / 2;
    }

    void next_shift(std::vector<double>& shift) {
        while (is_intersect(shift)) {
            for(int i = 0; i < number_of_dimensions; ++i)
                shift[i] = rand() % max_dim_size;
        }
        return;
    }

    std::vector<double> get_random_direction(const std::vector<double>&
                                             bad_direction = {0}) {
        std::vector<double> direction(number_of_dimensions);
        int sum_dir=0;
        for(double& dir : direction) {
            dir = rand() % 100;
            sum_dir += dir*dir;
        }
        sum_dir = std::sqrt(sum_dir);
        if (sum_dir == 0)
            return direction;
        for (double& dir : direction)
            dir /= sum_dir * std::pow(-1, rand() % 2);
        return direction;
    }

    int get_count_of_components() {
        std::vector<int> valid(number_of_classes, 0);
        int count = 0;
        for(int i=0;i<number_of_classes; ++i)
            if(!valid[i]){
            count++;
            dfs(i, valid);
        }
        return count;
    }

    void dfs(int i, std::vector<int>& valid) {
        valid[i] = 1;
        for (std::pair<int,double>& j : intersection[i])
            if (!valid[j.first])
                dfs(j.first, valid);
        return;
    }

    int add_intersect(std::string tmp_string) {
        std::vector<double> vals;
        get_vals(tmp_string, vals);
        if(vals[2] > 1 || vals[2] < 0)
            throw std::runtime_error("Incorrect intersection area");
        if (vals[2] == 0)
            return 0;
        intersection[vals[0] - 1].push_back({vals[1] - 1, vals[2]});
        intersection[vals[1] - 1].push_back({vals[0] - 1, vals[2]});
        return  0;
    }

    int add_class_count(std::string tmp_string) {
        number_of_elements.push_back(std::stoi(tmp_string));
        return 0;
    }

    int get_vals(std::string& tmp_string,
                 std::vector<int>& tmp_vector) {
        std::stringstream ss(tmp_string);
        tmp_vector.clear();
        int k = 0;
        while (ss >> k)
            tmp_vector.push_back(k);
        return 0;
    }

    int get_vals(std::string& tmp_string,
                 std::vector<double>& tmp_vector) {
        std::stringstream ss(tmp_string);
        tmp_vector.clear();
        double k = 0;
        while (ss >> k)
               tmp_vector.push_back(k);
        return 0;
    }

    std::vector<std::vector<std::pair<int, double>>> intersection;
    std::vector<int> number_of_elements;
    int number_of_classes;
    int number_of_dimensions;
    int fill = false;
    std::vector<std::vector<double>> not_valid;
    std::vector<std::vector<std::vector<double>>> data;
};

int main(int argc, char **argv) {
    std::string config = argv[1];
    Generator generator(config);
    generator.generate();
    generator.print(argv[2]);
}
