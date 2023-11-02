#include <fstream>
#include <sstream>
int main()
{
    const char nmfile[] = "out.txt";
    std::ofstream outseis(nmfile); // output, normal file

    for (int jj=0; jj<4000; jj++)
    {
            std::stringstream buf;
            for (int ii=0; ii<4000; ii++)
            {
                    int ij = jj + ii;
                    buf<<ij<<" ";
            }
            outseis << buf.str() << "\n";

    }
    outseis.close();
}