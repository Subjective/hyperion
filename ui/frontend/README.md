# Hyperion Dashboard Frontend

React-based web interface for visualizing and monitoring Hyperion optimization experiments. Built with Vite, TypeScript, and Tailwind CSS.

## Running

From the project root:

```bash
mise run ui-frontend
```

Or from this directory:

```bash
pnpm dev
```

## Building for Production

```bash
pnpm build
```

The built files will be in the `dist` directory.

## Tech Stack

- React 19 with TypeScript
- Vite for fast development
- Tailwind CSS for styling
- Framer Motion for animations
- Tanstack Query for data fetching
- Sonner for toast notifications
- Recharts for data visualization
- Zustand for state management

The frontend connects to the backend API on port 8000 and runs on port 5173 by default.
