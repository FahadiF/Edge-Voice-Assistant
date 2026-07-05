import { useEffect } from "react";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { ThemeProvider, applyUiSettings } from "./theme/ThemeProvider";
import { startWebSocket } from "./ws/socket";
import { settings } from "./api/endpoints";
import { Dashboard } from "./pages/Dashboard";
import { Conversation } from "./pages/Conversation";
import { Memory } from "./pages/Memory";
import { Personas } from "./pages/Personas";
import { Users } from "./pages/Users";
import { Models } from "./pages/Models";
import { Voices } from "./pages/Voices";
import { SettingsPage } from "./pages/Settings";
import { Diagnostics } from "./pages/Diagnostics";
import { Plugins } from "./pages/Plugins";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 10_000 },
  },
});

/** Applies persisted UI settings (theme/scale/motion) once loaded. */
function UiSettingsSync() {
  const { data } = useQuery({ queryKey: ["settings"], queryFn: settings.get });
  useEffect(() => {
    if (data) applyUiSettings(data.ui);
  }, [data]);
  return null;
}

export function App() {
  useEffect(() => {
    startWebSocket();
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <UiSettingsSync />
        <BrowserRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<Dashboard />} />
              <Route path="conversation" element={<Conversation />} />
              <Route path="memory" element={<Memory />} />
              <Route path="personas" element={<Personas />} />
              <Route path="users" element={<Users />} />
              <Route path="models" element={<Models />} />
              <Route path="voices" element={<Voices />} />
              <Route path="settings" element={<SettingsPage />} />
              <Route path="diagnostics" element={<Diagnostics />} />
              <Route path="plugins" element={<Plugins />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
