/* This file is part of GEGL
 *
 * GEGL is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * GEGL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GEGL; if not, see <http://www.gnu.org/licenses/>.
 *
 * Copyright 2003-2007 Calvin Williamson, Øyvind Kolås
 */

#include "config.h"
#define __GEGL_INIT_C

#include <babl/babl.h>

#include <glib-object.h>
#include <glib/gstdio.h>
#include <glib/gi18n-lib.h>

#include <stdlib.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef G_OS_WIN32
#include <process.h>
#endif

#include <gegl-debug.h>


guint gegl_debug_flags = 0; 


#include "gegl-instrument.h"
#include "gegl-init.h"
#include "module/geglmodule.h"
#include "module/geglmoduledb.h"
#include "gegl-types.h"
#include "buffer/gegl-buffer.h"
#include "operation/gegl-operation.h"
#include "operation/gegl-operations.h"
#include "operation/gegl-extension-handler.h"
#include "buffer/gegl-buffer-private.h"
#include "gegl-config.h"


/* if this function is made to return NULL swapping is disabled */
const gchar *
gegl_swap_dir (void)
{
  static gchar *swapdir = "";

  if (swapdir && swapdir[0] == '\0')
    {
      if (g_getenv ("GEGL_SWAP"))
        {
          if (g_str_equal (g_getenv ("GEGL_SWAP"), "RAM"))
            swapdir = NULL;
          else
            swapdir = g_strdup (g_getenv ("GEGL_SWAP"));
        }
      else
        {
          swapdir = g_build_filename (g_get_home_dir(),
                                      "." GEGL_LIBRARY,
                                      "swap",
                                      NULL);
        }

      /* Fall back to "swapping to RAM" if not able to create swap dir
       */
      if (swapdir &&
          ! g_file_test (swapdir, G_FILE_TEST_IS_DIR) &&
          g_mkdir_with_parents (swapdir, S_IRUSR | S_IWUSR | S_IXUSR) != 0)
        {
#if 0
          gchar *name = g_filename_display_name (swapdir);
          g_warning ("unable to create swapdir '%s': %s",
                     name, g_strerror (errno));
          g_free (name);
#endif

          swapdir = NULL;
        }
    }
  return swapdir;
};

static gboolean  gegl_post_parse_hook (GOptionContext *context,
                                       GOptionGroup   *group,
                                       gpointer        data,
                                       GError        **error);


static GeglConfig   *config = NULL;


static GeglModuleDB *module_db   = NULL;

static glong         global_time = 0;

static const gchar *makefile (void);

/**
 * gegl_init:
 * @argc: a pointer to the number of command line arguments.
 * @argv: a pointer to the array of command line arguments.
 *
 * Call this function before using any other GEGL functions. It will initialize
 * everything needed to operate GEGL and parses some standard command line
 * options.  @argc and @argv are adjusted accordingly so your own code will
 * never see those standard arguments.
 *
 * Note that there is an alternative ways to initialize GEGL: if you are
 * calling g_option_context_parse() with the option group returned by
 * gegl_get_option_group(), you don't have to call gegl_init().
 **/
void
gegl_init (gint    *argc,
           gchar ***argv)
{
  GOptionContext *context;
  GError         *error = NULL;
  if (config)
    return;

#if ENABLE_MP
  if (!g_thread_supported())
    g_thread_init (NULL);
#endif

  /*  If any command-line actions are ever added to GEGL, then the commented
   *  out code below should be used.  Until then, we simply call the parse hook
   *  directly.
   */
#if 0
  gegl_post_parse_hook (NULL, NULL, NULL, NULL);
#else

  context = g_option_context_new (NULL);
  g_option_context_set_ignore_unknown_options (context, TRUE);
  g_option_context_set_help_enabled (context, FALSE);
  g_option_context_set_main_group (context, gegl_get_option_group ());

  if (!g_option_context_parse (context, argc, argv, &error))
    {
      g_warning ("%s", error->message);
      g_error_free (error);
    }

  g_option_context_free (context);
#endif
}

static gchar   *cmd_gegl_swap=NULL;
static gchar   *cmd_gegl_cache_size=NULL;
static gchar   *cmd_gegl_quality=NULL;
static gchar   *cmd_babl_error=NULL;
static gboolean cmd_no_node_caches=FALSE;

static const GOptionEntry cmd_entries[]=
{
    {
     "babl-error", 0, 0,
     G_OPTION_ARG_STRING, &cmd_babl_error, 
     N_("babls error tolerance, a value beteen 0.2 and 0.000000001"), "<float>"
    },
    {
     "gegl-swap", 0, 0,
     G_OPTION_ARG_STRING, &cmd_gegl_swap, 
     N_("Where GEGL stores it's swap"), "<uri>"
    },
    {
     "gegl-cache-size", 0, 0, 
     G_OPTION_ARG_STRING, &cmd_gegl_cache_size, 
     N_("How much memory to (approximately) use for caching imagery"), "<megabytes>"
    },
    {
     "gegl-no-node-caches", 0, 0, 
     G_OPTION_ARG_NONE, &cmd_no_node_caches, 
     N_("Don't use per node caches to speed up _re_evaluation of the graph"), 
    },
    {
     "gegl-quality", 0, 0, 
     G_OPTION_ARG_STRING, &cmd_gegl_quality, 
     N_("The quality of rendering a value between 0.0(fast) and 1.0(reference)"), "<quality>"
    },
    { NULL }
};

/**
 * gegl_get_option_group:
 *
 * Returns a #GOptionGroup for the commandline arguments recognized
 * by GEGL. You should add this group to your #GOptionContext
 * with g_option_context_add_group(), if you are using
 * g_option_context_parse() to parse your commandline arguments.
 *
 * Returns a #GOptionGroup for the commandline arguments recognized by GEGL.
 */
GOptionGroup *
gegl_get_option_group (void)
{
  GOptionGroup *group;

  group = g_option_group_new ("gegl", "GEGL Options", "Show GEGL Options",
                              NULL, NULL);
  g_option_group_add_entries (group, cmd_entries);

  g_option_group_set_parse_hooks (group, NULL, gegl_post_parse_hook);

  return group;
}

GObject *gegl_config (void);
GObject *gegl_config (void)
{
  return G_OBJECT (config);
}

void gegl_tile_backend_ram_stats (void);
void gegl_tile_backend_tiledir_stats (void);
void gegl_tile_backend_file_stats (void);

void
gegl_exit (void)
{
  glong timing = gegl_ticks ();

  gegl_tile_cache_destroy ();
  gegl_operation_gtype_cleanup ();
  gegl_extension_handler_cleanup ();

  if (module_db != NULL)
    {
      g_object_unref (module_db);
      module_db = NULL;
    }

  babl_destroy ();

  timing = gegl_ticks () - timing;
  gegl_instrument ("gegl", "gegl_exit", timing);

  /* used when tracking buffer and tile leaks */
  if (g_getenv ("GEGL_DEBUG_BUFS") != NULL)
    {
      gegl_buffer_stats ();
      gegl_tile_backend_ram_stats ();
      gegl_tile_backend_file_stats ();
      gegl_tile_backend_tiledir_stats ();
    }
  global_time = gegl_ticks () - global_time;
  gegl_instrument ("gegl", "gegl", global_time);

  if (g_getenv ("GEGL_DEBUG_TIME") != NULL)
    {
      g_print ("\n%s", gegl_instrument_utf8 ());
    }

  if (gegl_buffer_leaks ())
    g_print ("  buffer-leaks: %i", gegl_buffer_leaks ());
  gegl_tile_cache_destroy ();

  if (gegl_swap_dir())
    {
      /* remove all files matching <$GEGL_SWAP>/GEGL-<pid>-*.swap */

      guint         pid     = getpid ();
      GDir         *dir     = g_dir_open (gegl_swap_dir (), 0, NULL);

      gchar        *glob    = g_strdup_printf ("GEGL-%i-*.swap", pid);
      GPatternSpec *pattern = g_pattern_spec_new (glob);
      g_free (glob);

      if (dir != NULL)
        {
          const gchar *name;

          while ((name = g_dir_read_name (dir)) != NULL)
            {
              if (g_pattern_match_string (pattern, name))
                {
                  gchar *fname = g_build_filename (gegl_swap_dir (),
                                                   name,
                                                   NULL);
                  g_unlink (fname);
                  g_free (fname);
                }
            }

          g_dir_close (dir);
        }

      g_pattern_spec_free (pattern);
    }
  g_object_unref (config);
  config = NULL;

  g_print ("\n");
}



static void
gegl_init_i18n (void)
{
  bindtextdomain (GETTEXT_PACKAGE, GEGL_LOCALEDIR);
  bind_textdomain_codeset (GETTEXT_PACKAGE, "UTF-8");
}

void     
gegl_get_version (int *major,   
                  int *minor,   
                  int *micro)   
{       
  if (major != NULL)    
    *major = GEGL_MAJOR_VERSION;        
                 
  if (minor != NULL)    
    *minor = GEGL_MINOR_VERSION;        
                 
  if (micro != NULL)    
    *micro = GEGL_MICRO_VERSION;        
}


static gboolean
gegl_post_parse_hook (GOptionContext *context,
                      GOptionGroup   *group,
                      gpointer        data,
                      GError        **error)
{
  glong time;

  if (config)
    return TRUE;


  g_assert (global_time == 0);
  global_time = gegl_ticks ();
  g_type_init ();
  gegl_instrument ("gegl", "gegl_init", 0);

  config = g_object_new (GEGL_TYPE_CONFIG, NULL);
  if (g_getenv ("GEGL_QUALITY"))
    config->quality = atof(g_getenv("GEGL_QUALITY")); 
  if (g_getenv ("GEGL_CACHE_SIZE"))
    config->cache_size = atoi(g_getenv("GEGL_CACHE_SIZE"))* 1024*1024; 

  config->node_caches = !cmd_no_node_caches;
  if (g_getenv ("GEGL_NO_NODE_CACHES"))
    g_object_set (config, "node-caches", FALSE, NULL);
  if (gegl_swap_dir())
    config->swap = g_strdup(gegl_swap_dir ());
  if (cmd_gegl_swap)
    g_object_set (config, "swap", cmd_gegl_swap, NULL);
  if (cmd_gegl_quality)
    config->quality = atof (cmd_gegl_quality);
  if (cmd_gegl_cache_size)
    config->cache_size = atoi (cmd_gegl_cache_size)*1024*1024;
  if (cmd_babl_error)
    g_object_set (config, "babl-error", atof(cmd_babl_error), NULL);

#ifdef GEGL_ENABLE_DEBUG
  {
    const char *env_string;
    env_string = g_getenv ("GEGL_DEBUG");
    if (env_string != NULL)
      {
        gegl_debug_flags =
          g_parse_debug_string (env_string,
                                gegl_debug_keys,
                                G_N_ELEMENTS (gegl_debug_keys));
        env_string = NULL;
      }
  }
#endif /* GEGL_ENABLE_DEBUG */

  time = gegl_ticks ();

  babl_init ();
  gegl_instrument ("gegl_init", "babl_init", gegl_ticks () - time);

  gegl_init_i18n ();

  time = gegl_ticks ();
  if (!module_db)
    {
      const gchar *gegl_path = g_getenv ("GEGL_PATH");

      module_db = gegl_module_db_new (FALSE);

      if (gegl_path)
        {
          gegl_module_db_load (module_db, gegl_path);
        }
      else
        {
          gchar *module_path;

#ifdef G_OS_WIN32
          module_path =
            g_win32_get_package_installation_subdirectory (NULL,
                                                           "lib" GEGL_LIBRARY "-0.dll",
                                                           "lib/" GEGL_LIBRARY);
#else
          module_path = g_build_filename (LIBDIR, GEGL_LIBRARY, NULL);
#endif

          gegl_module_db_load (module_db, module_path);
          g_free (module_path);

          /* also load plug-ins from ~/.gegl-0.0/plug-ins */
          module_path = g_build_filename (g_get_home_dir (),
                                          "." GEGL_LIBRARY,
                                          "plug-ins",
                                          NULL);

          if (g_mkdir_with_parents (module_path,
                                    S_IRUSR | S_IWUSR | S_IXUSR) == 0)
            {
              gchar *makefile_path = g_build_filename (module_path,
                                                       "Makefile",
                                                       NULL);

              if (! g_file_test (makefile_path, G_FILE_TEST_EXISTS))
                g_file_set_contents (makefile_path, makefile (), -1, NULL);

              g_free (makefile_path);
            }

          gegl_module_db_load (module_db, module_path);
          g_free (module_path);
        }

      gegl_instrument ("gegl_init", "load modules", gegl_ticks () - time);
    }

  gegl_instrument ("gegl", "gegl_init", gegl_ticks () - global_time);

  if (g_getenv ("GEGL_SWAP"))
    g_object_set (config, "swap", g_getenv ("GEGL_SWAP"), NULL);
  if (g_getenv ("GEGL_QUALITY"))
    {
      const gchar *quality = g_getenv ("GEGL_QUALITY");
      if (g_str_equal (quality, "fast"))
        g_object_set (config, "quality", 0.0, NULL);
      if (g_str_equal (quality, "good"))
        g_object_set (config, "quality", 0.5, NULL);
      if (g_str_equal (quality, "best"))
        g_object_set (config, "quality", 1.0, NULL);
    }

  return TRUE;
}



#ifdef GEGL_ENABLE_DEBUG
#if 0
static gboolean
gegl_arg_debug_cb (const char *key,
                   const char *value,
                   gpointer    user_data)
{
  gegl_debug_flags |=
    g_parse_debug_string (value,
                          gegl_debug_keys,
                          G_N_ELEMENTS (gegl_debug_keys));
  return TRUE;
}

static gboolean
gegl_arg_no_debug_cb (const char *key,
                      const char *value,
                      gpointer    user_data)
{
  gegl_debug_flags &=
    ~g_parse_debug_string (value,
                           gegl_debug_keys,
                           G_N_ELEMENTS (gegl_debug_keys));
  return TRUE;
}
#endif
#endif

/*
 * gegl_get_debug_enabled:
 *
 * Check if gegl has debugging turned on.
 *
 * Return value: TRUE if debugging is turned on, FALSE otherwise.
 */
gboolean
gegl_get_debug_enabled (void)
{
#ifdef GEGL_ENABLE_DEBUG
  return gegl_debug_flags != 0;
#else
  return FALSE;
#endif
}


static const gchar *makefile (void)
{
  return
    "# This is a generic makefile for GEGL operations. Just add .c files,\n"
    "# rename mentions of the filename and opname to the new name, and it should \n"
    "# compile. Operations in this dir should be loaded by GEGL by default\n"
    "# If the operation being written depends on extra libraries, you'd better\n"
    "# add a dedicated target with the extra bits linked in.\n"
    "\n\n"
    "CFLAGS  += `pkg-config gegl --cflags`  -I. -fPIC\n"
    /*"LDFLAGS += `pkg-config --libs` -shared\n"*/
    "SHREXT=.so\n"
    "CFILES = $(wildcard ./*.c)\n"
    "SOBJS  = $(subst ./,,$(CFILES:.c=$(SHREXT)))\n"
    "all: $(SOBJS)\n"
    "%$(SHREXT): %.c $(GEGLHEADERS)\n"
    "	@echo $@; $(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDADD)\n"
    "clean:\n"
    "	rm -f *$(SHREXT) $(OFILES)\n";
}


