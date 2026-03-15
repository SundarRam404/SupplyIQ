import './globals.css'

export const metadata = {
  title: 'SupplyIQ — AI Supply Chain Intelligence',
  description: 'A premium recruiter-ready website for demand forecasting, supply risk detection, contract intelligence, and executive supply chain decision support.',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
