/* This file is part of GEGL.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Copyright 2006 Øyvind Kolås <pippin@gimp.org>
 */
#include <glib.h>
#include <glib/gprintf.h>
#include <glib/gstdio.h>
#include "gegl-handlers.h"
#include "gegl-handler-cache.h"

G_DEFINE_TYPE (GeglHandlers, gegl_handlers, GEGL_TYPE_TILE_TRAIT)
static GObjectClass * parent_class = NULL;

static void
gegl_handlers_rebind (GeglHandlers *handlers);

static void
gegl_handlers_nuke_cache (GeglHandlers *handlers)
{
  GSList *iter;

  while (gegl_handlers_get_first (handlers, GEGL_TYPE_HANDLER_CACHE))
    {
      iter = handlers->chain;
      while (iter)
        {
          if (GEGL_IS_HANDLER_CACHE (iter->data))
            {
              g_object_unref (iter->data);
              handlers->chain = g_slist_remove (handlers->chain, iter->data);
              gegl_handlers_rebind (handlers);
              break;
            }
          iter = iter->next;
        }
    }
}

static void
dispose (GObject *object)
{
  GeglHandlers *handlers = GEGL_HANDLERS (object);
  GSList       *iter;

  /* Get rid of the cache before any further parts of the deconstruction of the
   * TileStore chain, unwritten tiles need a living TileStore for their
   * deconstruction.
   */
  gegl_handlers_nuke_cache (handlers);

  iter = handlers->chain;
  while (iter)
    {
      if (iter->data)
        g_object_unref (iter->data);
      iter = iter->next;
    }

  if (handlers->chain)
    g_slist_free (handlers->chain);
  handlers->chain = NULL;

  (*G_OBJECT_CLASS (parent_class)->dispose)(object);
}


static void
finalize (GObject *object)
{
  (*G_OBJECT_CLASS (parent_class)->finalize)(object);
}

static GeglTile *
get_tile (GeglTileStore *tile_store,
          gint           x,
          gint           y,
          gint           z)
{
  GeglHandlers   *handlers = GEGL_HANDLERS (tile_store);
  GeglTileStore  *source = GEGL_HANDLER (tile_store)->source;
  GeglTile       *tile   = NULL;

  if (handlers->chain != NULL)
    tile = gegl_tile_store_get_tile (GEGL_TILE_STORE (handlers->chain->data),
                                     x, y, z);
  else if (source)
    tile = gegl_tile_store_get_tile (source, x, y, z);
  else
    g_assert (0);
  return tile;
}

static gboolean
message (GeglTileStore  *tile_store,
         GeglTileMessage message,
         gint            x,
         gint            y,
         gint            z,
         gpointer        data)
{
  GeglHandlers *handlers = GEGL_HANDLERS (tile_store);
  GeglTileStore  *source = GEGL_HANDLER (tile_store)->source;

  if (handlers->chain != NULL)
    return gegl_tile_store_message (GEGL_TILE_STORE (handlers->chain->data), message, x, y, z, data);
  else if (source)
    return gegl_tile_store_message (source, message, x, y, z, data);
  else
    g_assert (0);

  return FALSE;
}

static void
gegl_handlers_class_init (GeglHandlersClass *class)
{
  GObjectClass       *gobject_class;
  GeglTileStoreClass *tile_store_class;

  gobject_class    = (GObjectClass *) class;
  tile_store_class = (GeglTileStoreClass *) class;

  tile_store_class->get_tile = get_tile;
  tile_store_class->message  = message;

  parent_class            = g_type_class_peek_parent (class);
  gobject_class->finalize = finalize;
  gobject_class->dispose  = dispose;
}

static void
gegl_handlers_init (GeglHandlers *self)
{
  self->chain = NULL;
}

GeglTileStore *tsource = NULL;

static void
gegl_handlers_rebind (GeglHandlers *handlers)
{
  GSList *iter;


  iter = handlers->chain;
  while (iter)
    {
      GeglHandler   *handler;
      GeglTileStore *source = NULL;

      handler = iter->data;
      if (iter->next)
        {
          source = g_object_ref (iter->next->data);
        }
      else
        {
          g_object_get (handlers, "source", &source, NULL);
        }
      g_object_set (G_OBJECT (handler), "source", source, NULL);
      g_object_unref (source);
      iter = iter->next;
    }
}

GeglHandler *
gegl_handlers_add (GeglHandlers *handlers,
                   GeglHandler  *handler)
{
  tsource       = GEGL_TILE_STORE (GEGL_HANDLER (handlers)->source);
  handlers->chain = g_slist_prepend (handlers->chain, handler);
  gegl_handlers_rebind (handlers);
  tsource = NULL;
  return handler;
}

/*
 * return the first handler of a given type
 */
GeglHandler *
gegl_handlers_get_first (GeglHandlers *handlers,
                         GType         type)
{
  GSList *iter;

  iter = handlers->chain;
  while (iter)
    {
      if ((G_TYPE_CHECK_INSTANCE_TYPE ((iter->data), type)))
        {
          return iter->data;
        }
      iter = iter->next;
    }
  return NULL;
}
