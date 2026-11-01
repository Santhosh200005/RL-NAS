import './globals.css';

export const metadata = {
  title: 'RL-NAS Dashboard',
  description: 'Reinforcement Learning–Driven Neural Architecture Search'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
