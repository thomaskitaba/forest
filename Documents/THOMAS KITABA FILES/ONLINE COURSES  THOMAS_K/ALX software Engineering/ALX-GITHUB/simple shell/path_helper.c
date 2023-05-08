#include "main.h"
/**
 * path_helper - add /bin/ as correction
 * @p_a: arg to be tokenized
 * @p_b: buffer to hold path
 * @d_len: default length
 * @p_len: path length
 * @n_c: need correction
 * Return: 1 on success, 0 on failur
 */
char *path_helper(char *p_a, char *p_b, int n_c, int d_len, int p_len)
{
char *default_path;
int i, k;

default_path = "/bin/";
k = 0;
if (n_c == 1)
{
/*append /bin/ and ls*/
strcpy(p_b, default_path);
for (i = d_len; i < d_len + p_len; i++)
{
p_b[i] = p_a[k];
k++;
}
p_b[i] = '\0';
return (p_b);
}
if (n_c == 0)
{
if (p_len <= 5)
return (NULL);
else
{
strcpy(p_b, p_a);
return (p_b);
}
}
return (NULL);
}
