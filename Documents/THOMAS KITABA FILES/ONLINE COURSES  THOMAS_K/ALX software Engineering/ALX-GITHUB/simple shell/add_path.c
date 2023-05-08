#include "main.h"
/**
 * add_path - add path to tokens index 0
 * @path_arg: path argument at index 0
 * @path_buf: buffer holding string
 * Return: append path return1,  path not appended, -1
 */
char *add_path(char *path_arg, char *path_buf)
{
char *def_path;
char *new_buf; 
int path_len, res, def_len, buffer_len, correction;
int i, k; /*TODO: to be deleted*/
path_len = buffer_len = k = 0;
def_path = "/bin/"; /*TODO: to be deleted*/
new_buf = NULL;
def_len = _strlen(def_path);
path_len = _strlen(path_arg);
correction = res = 0;
/*check if /bin/ exitst in path_arg*/
for (i = 0; i < def_len; i++)
{
if (def_path[i] != path_arg[i])
{
correction = 1;
break;
}
}
new_buf = path_helper(path_arg, path_buf, correction, def_len, path_len);
if (new_buf)
{
return (new_buf);
}
else
{
return (NULL);
}
}
