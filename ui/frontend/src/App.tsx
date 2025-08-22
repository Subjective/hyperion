import { Navigate, Outlet, RouterProvider, createBrowserRouter } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "@/api/client";
import Experiments from "./pages/Experiments";
import Experiment from "./pages/Experiment";

function Layout() {
  return <Outlet />;
}

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    errorElement: <ErrorBoundary />,
    children: [
      { index: true, element: <Navigate to="/experiments" replace /> },
      {
        path: "experiments",
        children: [
          { index: true, element: <Experiments /> },
          { path: ":id", element: <Experiment /> },
        ],
      },
    ],
  },
]);

function ErrorBoundary() {
  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100 mb-4">404</h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">Page not found</p>
        <a href="/experiments" className="text-blue-600 dark:text-blue-400 hover:underline">
          Go to Experiments
        </a>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}
