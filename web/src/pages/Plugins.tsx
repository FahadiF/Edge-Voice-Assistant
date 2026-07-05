/** Plugins (Part 12): manage discovered plugins (ADR-011). */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { plugins } from "../api/endpoints";
import { Card, EmptyState, toast } from "../components/common";

export function Plugins() {
  const queryClient = useQueryClient();
  const list = useQuery({ queryKey: ["plugins"], queryFn: plugins.list });

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ["plugins"] });

  const enable = useMutation({
    mutationFn: plugins.enable,
    onSuccess: () => { toast("success", "Plugin enabled"); invalidate(); },
    onError: (e) => toast("error", e.message),
  });
  const disable = useMutation({
    mutationFn: plugins.disable,
    onSuccess: () => { toast("info", "Plugin disabled"); invalidate(); },
    onError: (e) => toast("error", e.message),
  });
  const reload = useMutation({
    mutationFn: plugins.reload,
    onSuccess: () => { toast("success", "Plugin reloaded"); invalidate(); },
    onError: (e) => toast("error", e.message),
  });

  if (list.isLoading) return <p>Loading plugins…</p>;

  return (
    <div>
      <h1>Plugins</h1>
      <Card>
        {(list.data ?? []).length === 0 ? (
          <EmptyState>
            No plugins discovered. Plugins are installed as Python packages that register
            an <code>eva.plugins</code> entry point (ADR-011) — a marketplace for
            one-click install is planned for a future milestone.
          </EmptyState>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Version</th>
                <th>Description</th>
                <th>Contributes</th>
                <th>Permissions</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {(list.data ?? []).map((plugin) => (
                <tr key={plugin.id}>
                  <td>{plugin.name}</td>
                  <td>{plugin.version}</td>
                  <td>{plugin.description}</td>
                  <td>
                    {plugin.contributes.map((c) => (
                      <span key={c} className="chip">{c}</span>
                    ))}
                  </td>
                  <td>
                    {plugin.permissions.map((p) => (
                      <span key={p} className="chip chip-warning">{p}</span>
                    ))}
                  </td>
                  <td>
                    {!plugin.healthy && (
                      <span className="chip chip-danger" title={plugin.error ?? undefined}>
                        error
                      </span>
                    )}
                    <span className={`chip ${plugin.enabled ? "chip-success" : ""}`}>
                      {plugin.enabled ? "enabled" : "disabled"}
                    </span>
                  </td>
                  <td className="turn-actions">
                    {plugin.enabled ? (
                      <button onClick={() => disable.mutate(plugin.id)}>Disable</button>
                    ) : (
                      <button onClick={() => enable.mutate(plugin.id)}>Enable</button>
                    )}
                    <button onClick={() => reload.mutate(plugin.id)}>Reload</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}
