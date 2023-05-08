#include "main.h"
/**
 * _execve - execve
 * @av: 2D of CL arguments
 * @env: environment variable
 * Return: succes 1, failur -1.
 */
int _execve(char **av, char **env)
{
/*char *av_tem[] = {"/bin/ls", "-l", NULL};*/
int status;
struct stat st;

pid_t child_pid;
/* if path not found dont execute any thing*/
if (stat(av[0], &st) == 0)
child_pid = fork();
else
{
perror(av[0]);
return (-1);
}
if (child_pid == 0)
{
if (execve(av[0], av, env) == -1)
{
perror("Error insid execve:");
return (-1);
}
}
else if (child_pid > 0)
{
wait(&status);
return (1);
}
else
{
return (-1);
/*TODO: update exit struct*/
return (-1);
}
return (-1);
}
