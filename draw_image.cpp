#include <iostream>
using namespace std;
void drawImage(int image, Processor a, Processor b)
{
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            if ((int)(10 * (a.getPixel(image * 784 + i * 28 + j))) > 7)
            {
                cout << "O ";
            }
            else if ((int)(10 * (a.getPixel(image * 784 + i * 28 + j))) > 3)
            {
                cout << "o ";
            }
            else
            {
                cout << ". ";
            }
        }
        cout << endl;
    }
}