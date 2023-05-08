#include "main.h"
/**
 * main - entry point of the program
 * @argc: argument count
 * @argv: argument vector
 * @env: environment
 * Return: 0 on success
 */
int main(int argc, char **argv)
{
int mode; /*to accept returned value from hsh*/

if (argc <= 1 && !argv[1])
mode = 1; /*interactive*/
else
mode = 0; /*command line mode*/
hsh(argc, argv);
return (mode);
}
