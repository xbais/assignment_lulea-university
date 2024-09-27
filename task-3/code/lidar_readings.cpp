// lidar_readings.cpp

extern "C" {
    // Function to simulate LiDAR readings
    void lidar_readings(int x, int y, int grid_size, int** grid, int* readings) {
        // Up direction (reduce y)
        for (int i = y - 1; i >= 0; --i) {
            if (grid[i][x] == 1) { // 1 represents an obstacle
                readings[0] = y - i;
                break;
            }
        }

        // Down direction (increase y)
        for (int i = y + 1; i < grid_size; ++i) {
            if (grid[i][x] == 1) {
                readings[1] = i - y;
                break;
            }
        }

        // Left direction (reduce x)
        for (int i = x - 1; i >= 0; --i) {
            if (grid[y][i] == 1) {
                readings[2] = x - i;
                break;
            }
        }

        // Right direction (increase x)
        for (int i = x + 1; i < grid_size; ++i) {
            if (grid[y][i] == 1) {
                readings[3] = i - x;
                break;
            }
        }
    }
}
