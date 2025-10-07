#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

const int NUM_KNIGHTS = 4;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != NUM_KNIGHTS)
    {
        if (rank == 0)
        {
            std::cerr << "Ошибка: программу нужно запускать с " << NUM_KNIGHTS << " процессами." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::srand(static_cast<unsigned int>(time(0)) + rank);

    int elected = 0; // 0 - еще нет, 1 - старший выбран
    int attempts = 0;

    while (!elected)
    {
        attempts++;

        int my_vote = std::rand() % NUM_KNIGHTS;
        int vote_to_send;

        if (rank == 0)
        {
            vote_to_send = my_vote;
            MPI_Send(&vote_to_send, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

            int received_vote;
            MPI_Recv(&received_vote, 1, MPI_INT, NUM_KNIGHTS - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (received_vote == my_vote)
            {
                elected = 1;
                std::cout << "Chosen senior knight: knight #" << received_vote + 1 << std::endl;
                std::cout << "Total voting rounds: " << attempts << std::endl;
            }
            else
            {
                std::cout << "Round " << attempts << ": FAIL" << std::endl;
            }
        }
        else
        {
            int received_vote;
            MPI_Recv(&received_vote, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            vote_to_send = (received_vote == my_vote) ? my_vote : -1;

            int next_rank = (rank + 1) % NUM_KNIGHTS;
            MPI_Send(&vote_to_send, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
        }

        // синхронизация и проверка завершения
        MPI_Bcast(&elected, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
