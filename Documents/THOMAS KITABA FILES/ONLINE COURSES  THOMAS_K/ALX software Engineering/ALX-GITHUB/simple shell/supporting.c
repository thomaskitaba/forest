#include "main.h"
/**
 * _print_2D - print 2D array
 * @av: argument vector
 * @argc: argument count
 * Return: void
 */
void _print_2D(char **av, int argc)
{
int i;
for (i = 0; i < argc - 1; i++)
{
printf("%s\t", av[i]);
}
}
